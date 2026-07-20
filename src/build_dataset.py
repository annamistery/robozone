"""Шаг 3. Сборка YOLO-датасета и train/val-сплит + генерация conveyor.yaml.

Сплит идёт ПО ВИДЕО: соседние кадры одного ролика почти идентичны, и случайный
сплит по кадрам даёт утечку — валидация покажет mAP 0.98, а на реальной ленте
модель развалится.

    python -m src.build_dataset
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import yaml

from .common import ensure_dir, load_cfg, setup_logging

log = logging.getLogger("dataset")


def video_key(stem: str) -> str:
    """'clip_03_000120' -> 'clip_03' (имя кадра = <video_stem>_<frame_idx>)."""
    return stem.rsplit("_", 1)[0]


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--keep-empty", action="store_true", help="оставить кадры без объектов (фон)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    d = cfg["dataset"]
    frames = Path(cfg["paths"]["frames"])
    labels = Path(cfg["paths"]["labels_raw"])
    root = ensure_dir(cfg["paths"]["dataset"])

    pairs = []
    for img in sorted(frames.glob("*.jpg")):
        lbl = labels / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        if not args.keep_empty and lbl.stat().st_size == 0:
            continue
        pairs.append((img, lbl))

    if not pairs:
        log.error("Пар изображение+разметка не найдено. Сначала autolabel.")
        return

    groups: dict[str, list] = {}
    for img, lbl in pairs:
        groups.setdefault(video_key(img.stem) if d["split_by_video"] else img.stem, []).append((img, lbl))

    keys = sorted(groups)
    random.Random(d["seed"]).shuffle(keys)
    n_val = max(1, int(len(keys) * d["val_fraction"])) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])

    for split in ("train", "val"):
        ensure_dir(root / "images" / split)
        ensure_dir(root / "labels" / split)

    counts = {"train": 0, "val": 0}
    for k in keys:
        split = "val" if k in val_keys else "train"
        for img, lbl in groups[k]:
            shutil.copy2(img, root / "images" / split / img.name)
            shutil.copy2(lbl, root / "labels" / split / lbl.name)
            counts[split] += 1

    data_yaml = root / "conveyor.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": len(d["class_names"]),
                "names": d["class_names"],
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    log.info("train: %d кадров | val: %d кадров (%d видео в val)", counts["train"], counts["val"], n_val)
    log.info("data yaml -> %s", data_yaml)


if __name__ == "__main__":
    main()
