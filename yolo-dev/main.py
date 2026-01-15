#!/usr/bin/env python3
# train_yolo_roboflow_split.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def count_pairs(images_dir: Path, labels_dir: Path) -> Tuple[int, int, int]:
    """
    returns: (num_images, num_labels, num_pairs)
    pair = image with corresponding label (same stem + .txt)
    """
    imgs = list_images(images_dir)
    lbls = sorted([p for p in labels_dir.glob("*.txt") if p.is_file()]) if labels_dir.exists() else []
    lbl_set = {p.stem for p in lbls}

    pairs = 0
    for img in imgs:
        if img.stem in lbl_set:
            pairs += 1
    return len(imgs), len(lbls), pairs


def load_data_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"data.yaml not found: {path}")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Bad YAML format in {path}")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="Path to dataset root (contains data.yaml, train/, valid/, test/)")
    ap.add_argument("--model", default="yolo12s.pt", help="Pretrained weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0", help="GPU id like 0 or 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--use_test", action="store_true", help="Include test split in YAML (Ultralytics supports 'test:')")
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    yaml_in = data_dir / "data.yaml"

    # Roboflow splits
    train_img = data_dir / "train" / "images"
    train_lbl = data_dir / "train" / "labels"
    val_img   = data_dir / "valid" / "images"
    val_lbl   = data_dir / "valid" / "labels"
    test_img  = data_dir / "test" / "images"
    test_lbl  = data_dir / "test" / "labels"

    # sanity
    missing = []
    for p in [yaml_in, train_img, train_lbl, val_img, val_lbl]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit("Bad dataset structure. Missing:\n  " + "\n  ".join(missing))

    # counts (to catch issues early)
    tr_i, tr_l, tr_p = count_pairs(train_img, train_lbl)
    va_i, va_l, va_p = count_pairs(val_img, val_lbl)
    if tr_p < 2:
        raise SystemExit(f"Not enough labeled train pairs: {tr_p} (images={tr_i}, labels={tr_l})")
    if va_p < 1:
        raise SystemExit(f"Not enough labeled valid pairs: {va_p} (images={va_i}, labels={va_l})")

    te_i = te_l = te_p = 0
    have_test = args.use_test and test_img.exists() and test_lbl.exists()
    if have_test:
        te_i, te_l, te_p = count_pairs(test_img, test_lbl)
        if te_p < 1:
            have_test = False

    # read names from existing data.yaml (roboflow already put correct nc/names there)
    data_cfg = load_data_yaml(yaml_in)

    # enforce absolute base path + relative split paths (stable, reproducible)
    # robust for ultralytics: path + train/val/test relative to it
    names = data_cfg.get("names", None)
    nc = data_cfg.get("nc", None)
    if names is None or nc is None:
        raise SystemExit(f"{yaml_in} must contain 'nc' and 'names' (roboflow usually does).")

    yaml_out = data_dir / "dataset_ultralytics.yaml"
    out_text = (
        f"path: {data_dir}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        + (f"test: test/images\n" if have_test else "")
        + f"nc: {int(nc)}\n"
        f"names: {names}\n"
    )
    yaml_out.write_text(out_text, encoding="utf-8")

    print(f"[OK] dataset: {data_dir}")
    print(f"[OK] train: images={tr_i}, labels={tr_l}, pairs={tr_p}")
    print(f"[OK] valid: images={va_i}, labels={va_l}, pairs={va_p}")
    if have_test:
        print(f"[OK] test:  images={te_i}, labels={te_l}, pairs={te_p}")
    print(f"[OK] yaml: {yaml_out}")

    model = YOLO(args.model)
    model.train(
        data=str(yaml_out),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
