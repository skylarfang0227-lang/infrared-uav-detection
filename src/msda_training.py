"""Single-GPU protocol runner with full-precision state and no implicit evaluation."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.data.build import seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import init_seeds

from src.models.msda_irstd import MSDAIRSTD, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_state(model, optimizer, scaler, ema, scheduler, **metadata):
    return {"format": "msda_full_fp32_v1", "spec": model.spec, "enabled": model.enabled,
            "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "scaler": scaler.state_dict(), "ema": cpu_tree(ema.ema.state_dict()), "ema_updates": ema.updates,
            "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "gradients": {k: cpu_tree(p.grad) for k, p in model.named_parameters() if p.grad is not None}, **metadata}


def restore_state(state, model, optimizer, scaler, ema, scheduler):
    if state["spec"] != model.spec or state["enabled"] != model.enabled:
        raise ValueError("Checkpoint architecture differs from frozen model")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    ema.ema.load_state_dict(state["ema"], strict=True)
    ema.updates = state["ema_updates"]
    scheduler.load_state_dict(state["scheduler"])
    for name, param in model.named_parameters():
        param.grad = state["gradients"][name].to(param.device) if name in state["gradients"] else None
    restore_rng(state["rng"])


def atomic_checkpoint(state, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_model(path, device="cpu", ema=True):
    # Only locally produced, trusted checkpoints are supported.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "msda_full_fp32_v1":
        raise ValueError("Not a local MSDA full-state checkpoint")
    model = MSDAIRSTD(state["spec"], state["enabled"])
    model.load_state_dict(state["ema" if ema else "model"], strict=True)
    return model.to(device).float().eval(), state


class MSDATrainer(DetectionTrainer):
    """Reuse upstream loss/optimizer/data transforms, replace orchestration only.

    No validation loader is ever constructed by the trainer, even on the final
    epoch. Development COCO evaluation is an explicit call outside this class.
    """
    def __init__(self, phase, overrides, enabled=None):
        if phase not in {"smoke", "dev", "formal"}:
            raise ValueError(phase)
        self.phase, self.enabled = phase, enabled
        self.loader_audit = []
        super().__init__(overrides=overrides)
        self.callbacks = {k: [] for k in self.callbacks}  # no remote integrations
        self.next_epoch, self.last_opt_step = 0, -1
        self.history = []

    def get_dataset(self):
        import yaml
        data = yaml.safe_load(Path(self.args.data).read_text(encoding="utf-8"))
        expected = {"smoke": "smoke_train", "dev": "dev_train", "formal": "official_train"}[self.phase]
        train = ROOT / f"data/splits/sirst_uavb/{expected}.txt"
        val = ROOT / "data/splits/sirst_uavb/dev_val.txt"
        # Refuse test paths before check_det_dataset, image enumeration or cache access.
        if Path(data["train"]).resolve() != train.resolve() or Path(data["val"]).resolve() != val.resolve():
            raise ValueError("Training data must use phase-specific train and dev_val only")
        if "test" in data or "download" in data:
            raise ValueError("No test or download field is permitted")
        if data["names"] != {0: "UAV", 1: "Bird"}:
            raise ValueError("Class mapping mismatch")
        # Verify released member paths remain within the permitted processed train directory.
        allowed = (ROOT / "data/processed/sirst_uavb/images/train").resolve()
        evidence = json.loads((ROOT / "configs/dataset_evidence.json").read_text(encoding="utf-8"))
        if sha(train) != evidence["splits"][expected]["sha256"]:
            raise ValueError("Frozen split fingerprint changed")
        manifest = json.loads((ROOT / "data/splits/sirst_uavb/manifest.json").read_text(encoding="utf-8"))
        by_path = {(ROOT/r["image"]).resolve(): r for r in manifest["records"] if r["split"] == "train"}
        for line in train.read_text(encoding="utf-8").splitlines():
            if not Path(line).resolve().is_relative_to(allowed):
                raise ValueError("Unexpected image location in train members")
            image = Path(line).resolve()
            row = by_path[image]
            label = ROOT / "data/processed/sirst_uavb/labels/train" / (image.stem+".txt")
            if sha(image) != row["image_sha256"] or sha(label) != row["label_sha256"]:
                raise ValueError("Training image/label fingerprint changed")
        data.update(nc=2, channels=3, yaml_file=str(self.args.data))
        return data

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None or self.args.pretrained:
            raise ValueError("Scratch-only protocol")
        return MSDAIRSTD(enabled=self.enabled)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            self.loader_audit.append({"mode": mode, "action": "blocked_without_loading"})
            return []
        if Path(dataset_path).resolve() != Path(self.data["train"]).resolve():
            raise ValueError("Unexpected training split")
        self.loader_audit.append({"mode": mode, "path": str(dataset_path)})
        dataset = self.build_dataset(dataset_path, mode="train", batch=batch_size)
        return self.epoch_loader(dataset, batch_size, 0)

    def epoch_loader(self, dataset, batch, epoch):
        # Fresh workers/cache per epoch make epoch-boundary recovery deterministic.
        dataset.buffer = []
        dataset.ims = [None] * len(dataset)
        generator = torch.Generator().manual_seed(self.args.seed + 1000003*epoch)
        return DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=self.args.workers,
                          pin_memory=True, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                          generator=generator, persistent_workers=False)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

    def validate(self):
        raise RuntimeError("Use the separate frozen COCO evaluator")

    def final_eval(self):
        return None

    def setup(self):
        amp = self.args.amp
        self.args.amp = False  # Avoid upstream AMP probe downloading a different model.
        self._setup_train(world_size=1)
        self.args.amp = self.amp = amp
        torch.use_deterministic_algorithms(True, warn_only=False)
        # Default 65536 overflowed during the numeric smoke probe. Fix the scale
        # at one (no AP tuning); apply identically to reference and candidates.
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1., growth_interval=2147483647)
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                raise FloatingPointError("Invalid training gradient: "+name)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.)
        if not torch.isfinite(norm):
            raise FloatingPointError("Invalid gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)

    def save_model(self):
        state = capture_state(self.model, self.optimizer, self.scaler, self.ema, self.scheduler,
                              next_epoch=self.next_epoch, last_opt_step=self.last_opt_step,
                              train_args=vars(self.args), phase=self.phase, history=self.history,
                              implementation_sha256={p: sha(ROOT/p) for p in (
                                  "src/models/msda_irstd.py", "src/msda_training.py", "scripts/evaluate_sirst_protocol.py")},
                              train_split_sha256=sha(self.data["train"]),
                              protocol_sha256=sha(ROOT / "configs/protocols/sirst_msda_v1.json"))
        atomic_checkpoint(state, self.last)
        if self.next_epoch % self.args.save_period == 0:
            atomic_checkpoint(state, self.wdir / f"epoch{self.next_epoch}.pt")

    def resume_local(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["phase"] != self.phase or state["train_split_sha256"] != sha(self.data["train"]):
            raise ValueError("Resume dataset/phase mismatch")
        if state["protocol_sha256"] != sha(ROOT / "configs/protocols/sirst_msda_v1.json"):
            raise ValueError("Resume protocol changed")
        if not state.get("implementation_sha256"):
            raise ValueError("Checkpoint lacks implementation provenance")
        for name, digest in state["implementation_sha256"].items():
            if sha(ROOT/name) != digest:
                raise ValueError(f"Resume implementation changed: {name}")
        ignore = {"project", "name", "save_dir", "exist_ok", "model", "data"}
        for k, value in state["train_args"].items():
            if k not in ignore and getattr(self.args, k) != value:
                raise ValueError(f"Resume argument changed: {k}")
        restore_state(state, self.model, self.optimizer, self.scaler, self.ema, self.scheduler)
        self.next_epoch, self.last_opt_step = state["next_epoch"], state["last_opt_step"]
        self.history = state["history"]

    def run_epochs(self, stop_after=None, max_seconds=None, epoch_callback=None):
        started = time.monotonic()
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs*nb), 100) if self.args.warmup_epochs else -1
        for epoch in range(self.next_epoch, self.epochs):
            self.epoch = epoch
            if epoch >= self.epochs-self.args.close_mosaic:
                self.train_loader.dataset.close_mosaic(copy.copy(self.args))
            self.train_loader = self.epoch_loader(self.train_loader.dataset, self.batch_size, epoch)
            # Preserve the upstream epoch-indexed linear schedule, without a
            # spurious initial step-before-optimizer warning.
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
                self.scheduler.step()
            self.model.train()
            losses, batch_digests = [], []
            tick = time.monotonic()
            for i, batch in enumerate(self.train_loader):
                ni = i+nb*epoch
                if ni <= nw:
                    self.accumulate = max(1, int(np.interp(ni, [0, nw], [1, self.args.nbs/self.batch_size]).round()))
                    for j, group in enumerate(self.optimizer.param_groups):
                        group["lr"] = float(np.interp(ni, [0, nw], [self.args.warmup_bias_lr if j == 0 else 0., group["initial_lr"]*self.lf(epoch)]))
                        group["momentum"] = float(np.interp(ni, [0, nw], [self.args.warmup_momentum, self.args.momentum]))
                else:
                    self.accumulate = max(round(self.args.nbs/self.batch_size), 1)
                # SIRST is grayscale; ensure the three-channel input really is replicated gray.
                if not torch.equal(batch["img"][:, 0], batch["img"][:, 1]) or not torch.equal(batch["img"][:, 1], batch["img"][:, 2]):
                    raise ValueError("Non-replicated grayscale training input")
                if i == 0:
                    batch_digests.append(hashlib.sha256(batch["img"].numpy().tobytes()).hexdigest())
                batch = self.preprocess_batch(batch)
                with torch.autocast(device_type="cuda", enabled=self.amp):
                    loss, items = self.model(batch)
                    loss = loss.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                self.scaler.scale(loss).backward()
                if ni-self.last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.last_opt_step = ni
                losses.append(items.detach().cpu().tolist())
            self.next_epoch = epoch+1
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride"])
            self.history.append({"epoch": self.next_epoch, "loss": np.mean(losses, axis=0).tolist(),
                                 "seconds": time.monotonic()-tick, "first_batch_sha256": batch_digests[0]})
            if epoch_callback:
                epoch_callback(self)
            self.save_model()
            (self.save_dir / "progress.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            print(json.dumps(self.history[-1]), flush=True)
            if (stop_after is not None and self.next_epoch >= stop_after) or (max_seconds and time.monotonic()-started >= max_seconds):
                break
        return self.history


def protocol_overrides(phase, run_dir, seed=17, batch=8, epochs=None, workers=4):
    p = json.loads((ROOT / "configs/protocols/sirst_msda_v1.json").read_text(encoding="utf-8"))["training"]
    expected_epochs = p["epochs_formal" if phase == "formal" else "epochs_full_dev"]
    if phase != "smoke" and epochs is not None and epochs != expected_epochs:
        raise ValueError("Training epochs differ from the frozen local comparison budget")
    keys = ("optimizer", "lr0", "lrf", "cos_lr", "momentum", "weight_decay", "nbs", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
            "amp", "deterministic", "cache", "rect", "multi_scale", "mosaic", "close_mosaic", "scale", "translate", "fliplr", "flipud",
            "degrees", "shear", "perspective", "mixup", "copy_paste", "hsv_h", "hsv_s", "hsv_v")
    args = {k: p[k] for k in keys}
    train = {"smoke": "smoke_train", "dev": "dev_train", "formal": "official_train"}[phase]
    data = {"path": str(ROOT / "data/processed/sirst_uavb"), "train": str(ROOT / f"data/splits/sirst_uavb/{train}.txt"),
            "val": str(ROOT / "data/splits/sirst_uavb/dev_val.txt"), "names": {0: "UAV", 1: "Bird"}}
    import yaml
    run_dir = Path(run_dir)
    data_path = run_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    args.update(model=str(ROOT / "configs/models/yolo11n-p2.yaml"), data=str(data_path), device=0,
                batch=batch, epochs=epochs if epochs is not None else expected_epochs, workers=workers, seed=seed, imgsz=640,
                pretrained=False, patience=0, val=False, plots=False, save=True, save_period=50,
                box=7.5, cls=.5, dfl=1.5, project=str(run_dir), name="train", exist_ok=False)
    # The short resource test exercises both augmented and mosaic-closed training.
    if phase == "smoke":
        args["close_mosaic"] = 1
    return args
