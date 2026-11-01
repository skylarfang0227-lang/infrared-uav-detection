"""Validate and copy the supplied SIRST-UAVB ZIP; never inspect other datasets."""
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive = ROOT / "SIRST-UAVB.zip"
    expected = json.loads((ROOT / "data/splits/sirst_uavb/manifest.json").read_text(encoding="utf-8"))["archive_sha256"]
    assert digest(archive.read_bytes()) == expected, "Archive differs from the recorded snapshot"
    destination = ROOT / "data/processed/sirst_uavb"
    split_dir = ROOT / "data/splits/sirst_uavb"
    split_dir.mkdir(parents=True, exist_ok=True)
    records, issues = [], []
    counts = Counter()
    with zipfile.ZipFile(archive) as source:
        for split in ("train", "val"):
            ids = source.read(f"SIRST-UAVB/{split}.txt").decode("utf-8-sig").split()
            assert len(ids) == len(set(ids)), "Repeated split member"
            for folder in ("images", "labels"):
                (destination / folder / split).mkdir(parents=True, exist_ok=True)
            for sample_id in ids:
                assert sample_id.isdigit(), "Unsafe image ID"
                image_bytes = source.read(f"SIRST-UAVB/images/{split}/{sample_id}.jpg")
                label_bytes = source.read(f"SIRST-UAVB/labels/{split}/{sample_id}.txt")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = image.size
                    image.verify()
                boxes = []
                for line in label_bytes.decode("utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    values = [float(v) for v in line.split()]
                    assert len(values) == 5 and all(math.isfinite(v) for v in values)
                    category, x, y, w, h = values
                    assert category in (0, 1) and 0 <= x <= 1 and 0 <= y <= 1
                    assert 0 < w <= 1 and 0 < h <= 1
                    if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                        issues.append({"id": sample_id, "split": split, "issue": "box_outside_image", "box": values})
                    boxes.append(values)
                    counts[f"{split}/{int(category)}"] += 1
                image_path = destination / "images" / split / f"{sample_id}.jpg"
                label_path = destination / "labels" / split / f"{sample_id}.txt"
                for path, data in ((image_path, image_bytes), (label_path, label_bytes)):
                    if path.exists():
                        assert digest(path.read_bytes()) == digest(data), f"Existing output differs: {path}"
                    else:
                        path.write_bytes(data)
                records.append({"id": sample_id, "split": split, "image": image_path.relative_to(ROOT).as_posix(),
                                "image_sha256": digest(image_bytes), "label_sha256": digest(label_bytes),
                                "width": width, "height": height, "boxes": boxes})
    assert not issues, f"Invalid geometry, inspect before training: {issues[:5]}"
    train = [r for r in records if r["split"] == "train"]
    evaluation = [r for r in records if r["split"] == "val"]
    # Group exact duplicates, never use the official evaluation score for selection.
    dev = [r for r in train if int(digest(("dev-v1:"+r["image_sha256"]).encode())[:8], 16) % 5 == 0]
    dev_hashes = {r["image_sha256"] for r in dev}
    fit = [r for r in train if r["image_sha256"] not in dev_hashes]
    assert not dev_hashes.intersection(r["image_sha256"] for r in fit)
    assert len(train) == 2400 and len(evaluation) == 600
    groups = {"official_train": train, "official_eval": evaluation, "dev_train": fit, "dev_val": dev,
              "smoke_train": sorted(fit, key=lambda r: digest(("smoke:"+r["id"]).encode()))[:256],
              "smoke_val": sorted(dev, key=lambda r: digest(("smoke:"+r["id"]).encode()))[:64]}
    for name, rows in groups.items():
        (split_dir / f"{name}.txt").write_text("".join((ROOT / r["image"]).as_posix()+"\n" for r in rows), encoding="utf-8")
    config_dir = ROOT / "configs"
    config_dir.mkdir(exist_ok=True)
    for protocol, fit_name, val_name in (("smoke", "smoke_train", "smoke_val"), ("dev", "dev_train", "dev_val"),
                                         ("official", "official_train", "official_eval")):
        config = {"path": destination.as_posix(), "train": (split_dir / f"{fit_name}.txt").as_posix(),
                  "val": (split_dir / f"{val_name}.txt").as_posix(), "names": {0: "UAV", 1: "Bird"}}
        (config_dir / f"sirst_{protocol}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"archive_sha256": digest(archive.read_bytes()), "class_source": "https://github.com/JN-Yang/PConv-SDloss-Data/issues/6",
                "names": {0: "UAV", 1: "Bird"}, "box_counts": dict(counts), "split_counts": {k: len(v) for k, v in groups.items()},
                "published_box_counts": {"UAV": 2955, "Bird": 1742},
                "limitations": ["Local label counts differ from the paper; official cloud package identity unverified.",
                                "Dev split groups exact duplicates only; sequence identities and near-duplicate independence unavailable.",
                                "Official eval contains exact images seen in official training; explicitly authorized disclosed benchmark only."],
                "issues": issues, "records": records}
    (split_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
