"""Альтернативный рантайм строго по схеме с картинки: YOLO (боксы) -> MobileSAM (маска).

Когда это лучше, чем end-to-end YOLO-seg (infer_track.py):
  + маски заметно точнее по краям (SAM даёт пиксельные границы, а не 160x160 прототипы);
  + детектор можно дообучать, не трогая сегментатор.
Когда хуже:
  - MobileSAM это отдельный прогон энкодера на каждый кадр (~30-60 FPS на RTX 5080
    против ~200+ у yolo11n-seg), плюс декодер на каждый бокс.

Для конвейера с контрастным фоном обычно достаточно YOLO-seg. Этот модуль —
для случаев, когда робот промахивается из-за грубых краёв маски.

    python -m src.runtime_yolo_sam --source "C:/.../test.mp4" --show
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np
from ultralytics import SAM, YOLO

from .common import load_cfg, setup_logging

log = logging.getLogger("yolo+sam")


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--det-weights", required=True, help="веса YOLO-детектора (боксы)")
    ap.add_argument("--sam-weights", default="mobile_sam.pt")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    r = cfg["runtime"]

    det = YOLO(args.det_weights)
    sam = SAM(args.sam_weights)

    t0, n = time.time(), 0
    for res in det.track(source=args.source, stream=True, persist=True,
                         tracker=r["tracker"], conf=r["conf"], iou=r["iou"],
                         imgsz=r["imgsz"], verbose=False):
        frame = res.orig_img.copy()
        n += 1
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        ids = res.boxes.id.int().cpu().tolist() if (res.boxes is not None and res.boxes.id is not None) else []

        if len(boxes):
            sres = sam(res.orig_img, bboxes=boxes.tolist(), verbose=False)[0]
            if sres.masks is not None:
                overlay = frame.copy()
                for i, poly in enumerate(sres.masks.xy):
                    if len(poly) < 3:
                        continue
                    pts = poly.astype(np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 200, 120))
                    cv2.polylines(frame, [pts], True, (0, 200, 120), 2)
                    if i < len(ids):
                        M = cv2.moments(pts)
                        if M["m00"]:
                            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                            cv2.putText(frame, f"#{ids[i]}", (cx - 15, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        cv2.putText(frame, f"FPS {n / max(time.time() - t0, 1e-6):.1f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if args.show:
            cv2.imshow("yolo+sam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    log.info("Средний FPS: %.1f", n / max(time.time() - t0, 1e-6))


if __name__ == "__main__":
    main()
