"""Шаг 1. Извлечение кадров из сырых видео с конвейера.

Отсеиваем смазанные кадры и почти-дубликаты — иначе Grounding DINO
будет часами размечать сотни одинаковых картинок, а YOLO переобучится
на одном и том же товаре.

    python -m src.extract_frames
"""
from __future__ import annotations

import argparse
import logging

import cv2

from .common import (
    dhash,
    ensure_dir,
    hamming,
    list_videos,
    load_cfg,
    setup_logging,
    variance_of_laplacian,
)

log = logging.getLogger("extract")


def process_video(path, out_dir, cfg) -> int:
    fc = cfg["frames"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.warning("Не открылось: %s", path)
        return 0

    saved, idx, last_hashes = 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % fc["every_n"] != 0:
            continue
        if saved >= fc["max_per_video"]:
            break

        long_side = fc["resize_long_side"]
        if long_side and max(frame.shape[:2]) > long_side:
            s = long_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if variance_of_laplacian(gray) < fc["blur_threshold"]:
            continue

        h = dhash(gray)
        if any(hamming(h, prev) < fc["dedup_threshold"] for prev in last_hashes[-30:]):
            continue
        last_hashes.append(h)

        name = f"{path.stem}_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    log.info("%-40s -> %d кадров", path.name, saved)
    return saved


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_dir = ensure_dir(cfg["paths"]["frames"])
    videos = list_videos(cfg["paths"]["videos"])
    if not videos:
        log.error("Видео не найдены в %s", cfg["paths"]["videos"])
        return

    log.info("Найдено видео: %d", len(videos))
    total = sum(process_video(v, out_dir, cfg) for v in videos)
    log.info("Итого кадров: %d -> %s", total, out_dir)


if __name__ == "__main__":
    main()
