"""Шаг 2. Авто-дистилляция: Grounding DINO (текст -> боксы) + SAM (бокс -> маска).

На выходе — .txt в формате YOLO-seg, все объекты с class_id = 0 ('item').

    python -m src.autolabel                # разметить все кадры
    python -m src.autolabel --limit 50 --preview   # прогнать 50 кадров и посмотреть глазами

Модели тянутся один раз в кэш HuggingFace / ultralytics и дальше работают локально.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM

from .common import (
    draw_overlay,
    ensure_dir,
    load_cfg,
    mask_to_polygon,
    nms_boxes,
    polygon_to_yolo_line,
    setup_logging,
)

log = logging.getLogger("autolabel")


class GroundingDINO:
    def __init__(self, cfg: dict, device: str):
        g = cfg["gdino"]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(g["model_id"])
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(g["model_id"]).to(device).eval()
        )
        self.prompt = g["prompt"].lower().strip()
        self.box_thr = g["box_threshold"]
        self.text_thr = g["text_threshold"]

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_thr,
            text_threshold=self.text_thr,
            target_sizes=[pil.size[::-1]],  # (h, w)
        )[0]
        boxes = res["boxes"].cpu().numpy()          # xyxy
        scores = res["scores"].cpu().numpy()
        return boxes, scores


def filter_boxes(boxes, scores, w, h, g) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores
    frame_area = w * h
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (areas > g["min_box_area_frac"] * frame_area) & (areas < g["max_box_area_frac"] * frame_area)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return boxes, scores

    idx = nms_boxes(boxes, scores, g["nms_iou"])
    boxes, scores = boxes[idx], scores[idx]

    order = np.argsort(-scores)[: g["max_boxes_per_frame"]]
    return boxes[order], scores[order]


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="обработать только N кадров (отладка)")
    ap.add_argument("--preview", action="store_true", help="сохранять картинки с наложенными масками")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    frames_dir = Path(cfg["paths"]["frames"])
    labels_dir = ensure_dir(cfg["paths"]["labels_raw"])
    preview_dir = ensure_dir(cfg["paths"]["preview"]) if args.preview else None

    images = sorted(p for p in frames_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        log.error("Нет кадров в %s — сначала запусти extract_frames", frames_dir)
        return

    dino = GroundingDINO(cfg, device)
    sam = SAM(cfg["sam"]["weights"])

    s = cfg["sam"]
    empty, total_objs = 0, 0

    for i, img_path in enumerate(images, 1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        boxes, scores = dino(bgr)
        boxes, scores = filter_boxes(boxes, scores, w, h, cfg["gdino"])

        lines, polys = [], []
        if len(boxes):
            res = sam(bgr, bboxes=boxes.tolist(), verbose=False)[0]
            if res.masks is not None:
                for m in res.masks.data.cpu().numpy():
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    if m.sum() < s["min_mask_area_frac"] * w * h:
                        continue
                    poly = mask_to_polygon(m, s["polygon_epsilon"], s["max_polygon_points"])
                    if poly is None:
                        continue
                    polys.append(poly)
                    lines.append(polygon_to_yolo_line(poly, w, h, cls_id=0))

        (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        total_objs += len(lines)
        if not lines:
            empty += 1

        if preview_dir is not None:
            cv2.imwrite(str(preview_dir / img_path.name), draw_overlay(bgr, polys))

        if i % 25 == 0 or i == len(images):
            log.info("%d/%d кадров | объектов: %d | пустых: %d", i, len(images), total_objs, empty)

    log.info("Готово. Разметка -> %s", labels_dir)
    log.info("Среднее число объектов на кадр: %.2f", total_objs / max(len(images), 1))
    if empty / max(len(images), 1) > 0.3:
        log.warning(
            "Более 30%% кадров пустые. Понизь gdino.box_threshold или поправь prompt "
            "под то, что реально едет по ленте."
        )


if __name__ == "__main__":
    main()
