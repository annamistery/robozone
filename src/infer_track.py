"""Шаг 5. Рантайм: YOLO11-seg + ByteTrack, визуализация и выгрузка для робота.

    python -m src.infer_track --source "C:/.../video/test.mp4" --show
    python -m src.infer_track --source 0 --save-jsonl picks.jsonl

Каждому пятну присваивается стабильный track_id — робот понимает, что маска
на 10 кадрах подряд это один и тот же товар, и не пытается взять его дважды.
Событие 'pick' генерируется один раз на track_id при пересечении линии захвата.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .common import ensure_dir, load_cfg, setup_logging

log = logging.getLogger("runtime")


def color_for(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def resolve_weights(cfg: dict, override: str | None) -> str:
    if override:
        return override
    if cfg["runtime"]["weights"]:
        return cfg["runtime"]["weights"]
    best = Path(cfg["paths"]["runs"]) / cfg["train"]["name"] / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Не найдены веса: {best}. Укажи --weights.")
    return str(best)


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True, help="путь к видео, папке или индекс камеры")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--save-jsonl", default=None, help="лог событий для контроллера робота")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    r = cfg["runtime"]
    weights = resolve_weights(cfg, args.weights)
    log.info("Веса: %s", weights)

    model = YOLO(weights)
    source = int(args.source) if args.source.isdigit() else args.source

    writer = None
    jsonl = open(args.save_jsonl, "w", encoding="utf-8") if args.save_jsonl else None
    picked: set[int] = set()
    prev_cy: dict[int, float] = {}
    t0, n_frames = time.time(), 0

    stream = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=r["tracker"],
        conf=r["conf"],
        iou=r["iou"],
        imgsz=r["imgsz"],
        retina_masks=True,
        verbose=False,
    )

    for res in stream:
        frame = res.orig_img.copy()
        h, w = frame.shape[:2]
        line_y = int(r["pick_line_y"] * h)
        n_frames += 1

        cv2.line(frame, (0, line_y), (w, line_y), (0, 165, 255), 2)

        if res.masks is not None and res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            polys = res.masks.xy  # список (K,2) в пикселях исходного кадра

            overlay = frame.copy()
            for tid, conf, poly in zip(ids, confs, polys):
                if len(poly) < 3:
                    continue
                pts = poly.astype(np.int32)
                col = color_for(tid)
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(frame, [pts], True, col, 2)

                M = cv2.moments(pts)
                if M["m00"] == 0:
                    continue
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                area = float(cv2.contourArea(pts))

                cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)
                cv2.putText(frame, f"#{tid} {conf:.2f}", (int(cx) - 25, int(cy) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # событие захвата: центроид пересёк линию сверху вниз, один раз на ID
                if tid not in picked and prev_cy.get(tid, cy) < line_y <= cy:
                    picked.add(tid)
                    rec = {
                        "event": "pick",
                        "track_id": tid,
                        "frame": n_frames,
                        "ts": round(time.time() - t0, 3),
                        "centroid_px": [round(cx, 1), round(cy, 1)],
                        "centroid_norm": [round(cx / w, 5), round(cy / h, 5)],
                        "area_px": round(area, 1),
                        "conf": round(conf, 3),
                        "polygon_norm": [[round(x / w, 5), round(y / h, 5)] for x, y in poly.tolist()],
                    }
                    if jsonl:
                        jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        jsonl.flush()
                    log.info("PICK id=%d centroid=(%.0f, %.0f) area=%.0f", tid, cx, cy, area)
                prev_cy[tid] = cy

            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        fps = n_frames / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS {fps:.1f} | picked {len(picked)}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.save_video:
            if writer is None:
                ensure_dir(Path(args.save_video).parent)
                writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(frame)

        if args.show:
            cv2.imshow("conveyor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    if jsonl:
        jsonl.close()
    cv2.destroyAllWindows()
    log.info("Кадров: %d | средний FPS: %.1f | уникальных товаров: %d",
             n_frames, n_frames / max(time.time() - t0, 1e-6), len(picked))


if __name__ == "__main__":
    main()
