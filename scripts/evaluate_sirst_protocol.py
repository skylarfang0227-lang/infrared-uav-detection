"""Frozen FP32 SIRST evaluator; no training-time access to official evaluation."""
import argparse
import contextlib
import io
import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.msda_training import load_model, sha


def records_for_split(split="dev_val"):
    if split not in {"dev_val", "official_eval"}:
        raise ValueError(split)
    members = {Path(p).resolve() for p in (ROOT / f"data/splits/sirst_uavb/{split}.txt").read_text().splitlines()}
    manifest = json.loads((ROOT / "data/splits/sirst_uavb/manifest.json").read_text())
    records = [r for r in manifest["records"] if (ROOT / r["image"]).resolve() in members]
    if len(records) != len(members):
        raise ValueError("Manifest/member mismatch")
    return records


def score(records, predictions):
    images, annotations = [], []
    for r in records:
        images.append({"id": int(r["id"]), "width": r["width"], "height": r["height"]})
        for cls, x, y, w, h in r["boxes"]:
            x, y, w, h = x*r["width"], y*r["height"], w*r["width"], h*r["height"]
            if w <= 0 or h <= 0:
                raise ValueError("Invalid target box")
            annotations.append({"id": len(annotations)+1, "image_id": int(r["id"]), "category_id": int(cls)+1,
                                "bbox": [x-w/2, y-h/2, w, h], "area": w*h, "iscrowd": 0})
    ids = {x["id"] for x in images}
    if len(ids) != len(images):
        raise ValueError("Repeated image ID")
    for p in predictions:
        if p["image_id"] not in ids or p["category_id"] not in {1, 2} or not np.isfinite(p["bbox"]+[p["score"]]).all():
            raise ValueError("Invalid prediction")
    groups = {"all": [0., 1e10], "tiny": [np.nextafter(0., np.inf), 16.],
              "area_16to49": [np.nextafter(16., np.inf), 49.], "area_gt49": [np.nextafter(49., np.inf), 1e10]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"info": {}, "images": images, "annotations": annotations,
                      "categories": [{"id": 1, "name": "UAV"}, {"id": 2, "name": "Bird"}]}
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO(); dt.dataset = {**gt.dataset, "annotations": []}; dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.params.areaRng, ev.params.areaRngLbl = list(groups.values()), list(groups)
        ev.evaluate(); ev.accumulate()
    precision, metrics = ev.eval["precision"], {}
    def mean_valid(x):
        values = x[x >= 0]
        return float(values.mean()) if values.size else None
    for ai, (group, (lo, hi)) in enumerate(groups.items()):
        for ki, name in enumerate(("UAV", "Bird")):
            values = precision[:, :, ki, ai, -1]
            metrics[f"{name}/{group}"] = {
                "targets": int(sum(a["category_id"] == ki+1 and lo <= a["area"] <= hi for a in annotations)),
                "ap50_95": mean_valid(values), "ap50": mean_valid(values[0]), "ap75": mean_valid(values[5])}
    supported = [v["ap50_95"] for k, v in metrics.items() if k.startswith("UAV/") and k != "UAV/all" and v["targets"] >= 30]
    return {"primary": metrics["UAV/all"]["ap50_95"], "primary_key": "UAV/all/AP50:95",
            "metrics": metrics, "two_class_ap50_95": mean_valid(precision[:, :, :, 0, -1]),
            "two_class_ap50": mean_valid(precision[0, :, :, 0, -1]),
            "worst_supported_size_ap50_95": min(supported) if supported else None,
            "low_scr": None, "cross_scene": None, "cross_dataset": None,
            "evaluator": "pycocotools2.0.10 COCOeval bbox; maxDets[1,10,300]; native float boxes"}


def image_tensor(record, device):
    gray = cv2.imread(str(ROOT / record["image"]), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape[:2] != (record["height"], record["width"]):
        raise ValueError("Unreadable or wrong-sized image")
    # Ultralytics' cv2 patch retains the singleton channel for grayscale reads.
    if gray.ndim == 3 and gray.shape[2] == 1:
        gray = gray[:, :, 0]
    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    im = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=im)
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).to(device).float().unsqueeze(0)/255.


@torch.inference_mode()
def evaluate_model(model, output, split="dev_val", records=None, compress=False):
    if split != "dev_val":
        raise RuntimeError("Official evaluation requires the separately frozen formal assessment command")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    records = records if records is not None else records_for_split(split)
    model.eval().float()
    device = next(model.parameters()).device
    predictions = []
    start = time.monotonic()
    for record in records:
        x = image_tensor(record, device)
        pred = model(x)
        boxes = ops.non_max_suppression(pred, conf_thres=.001, iou_thres=.7, multi_label=False,
                                        agnostic=False, max_det=300, nc=2, max_time_img=10.)[0]
        ops.scale_boxes(x.shape[2:], boxes[:, :4], (record["height"], record["width"]))
        for b in boxes.cpu().tolist():
            predictions.append({"image_id": int(record["id"]), "category_id": int(b[5])+1,
                                "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "score": b[4]})
    predictions_path = output / ("predictions.json.gz" if compress else "predictions.json")
    if compress:
        with gzip.open(predictions_path, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(predictions, f, separators=(",", ":"))
    else:
        predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    result = score(records, predictions)
    result.update(image_count=len(records), elapsed_seconds=time.monotonic()-start,
                  predictions_sha256=sha(predictions_path), predictions_file=predictions_path.name, split=split)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint, "cuda")
    result = evaluate_model(model, args.output)
    print(json.dumps({"primary": result["primary"], "images": result["image_count"]}))


if __name__ == "__main__":
    main()
