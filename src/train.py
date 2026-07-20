"""Шаг 4. Обучение YOLO11-seg на авто-разметке (nc=1, names=['item']).

    python -m src.train
    python -m src.train --model yolo11s-seg.pt --epochs 150

Аугментации подкручены под конвейер: объекты едут в одной плоскости,
масштаб почти фиксирован, перспектива не меняется — сильный mosaic/scale
только вредит. Зато нужны flip и вариации яркости (блики на ленте).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

from .common import load_cfg, setup_logging

log = logging.getLogger("train")


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    t = cfg["train"]
    data_yaml = Path(cfg["paths"]["dataset"]) / "conveyor.yaml"
    if not data_yaml.exists():
        log.error("Нет %s — сначала build_dataset", data_yaml)
        return

    model = YOLO(args.model or t["model"])
    model.train(
        task="segment",
        data=str(data_yaml),
        epochs=args.epochs or t["epochs"],
        imgsz=t["imgsz"],
        batch=args.batch or t["batch"],
        device=t["device"],
        workers=t["workers"],
        patience=t["patience"],
        project=cfg["paths"]["runs"],
        name=t["name"],
        exist_ok=True,
        # --- аугментации под фиксированную камеру над лентой ---
        mosaic=0.3,
        close_mosaic=15,     # последние 15 эпох — без мозаики
        scale=0.25,
        degrees=10.0,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.5,          # вид сверху -> вертикальный флип валиден
        hsv_v=0.5,           # блики / неравномерная подсветка
        hsv_s=0.6,
        erasing=0.2,
        cos_lr=True,
        seed=cfg["dataset"]["seed"],
    )

    metrics = model.val()
    log.info("mask mAP50: %.4f | mask mAP50-95: %.4f", metrics.seg.map50, metrics.seg.map)
    log.info("Веса: %s", Path(cfg["paths"]["runs"]) / t["name"] / "weights" / "best.pt")


if __name__ == "__main__":
    main()
