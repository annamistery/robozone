"""Общие утилиты: конфиг, логирование, работа с масками/полигонами."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_cfg(path: str | Path = None) -> dict:
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "project.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_videos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Папка с видео не найдена: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT)


# ---------------------------------------------------------------- изображения

def variance_of_laplacian(gray: np.ndarray) -> float:
    """Метрика резкости. Низкое значение = смазанный кадр."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Перцептивный хэш для отсева почти одинаковых кадров."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in diff.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------- геометрия

def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Class-agnostic NMS. boxes: (N,4) xyxy."""
    if len(boxes) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
        scores=[float(s) for s in scores],
        score_threshold=0.0,
        nms_threshold=float(iou_thr),
    )
    if len(idxs) == 0:
        return []
    return np.array(idxs).flatten().tolist()


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_frac: float = 0.0015,
    max_points: int = 120,
) -> np.ndarray | None:
    """Бинарная маска (H,W) -> самый крупный контур как (K,2) в пикселях."""
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None
    eps = epsilon_frac * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    if len(approx) > max_points:  # равномерное прореживание
        keep = np.linspace(0, len(approx) - 1, max_points).astype(int)
        approx = approx[keep]
    return approx.astype(np.float32)


def polygon_to_yolo_line(poly: np.ndarray, w: int, h: int, cls_id: int = 0) -> str:
    """(K,2) пиксели -> строка YOLO-seg: 'cls x1 y1 x2 y2 ...' (нормализовано)."""
    p = poly.copy().astype(np.float64)
    p[:, 0] = np.clip(p[:, 0] / w, 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1] / h, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in p.flatten())
    return f"{cls_id} {coords}"


def draw_overlay(img: np.ndarray, polys: Iterable[np.ndarray], color=(0, 200, 120), alpha=0.45) -> np.ndarray:
    out = img.copy()
    layer = img.copy()
    for p in polys:
        cv2.fillPoly(layer, [p.astype(np.int32)], color)
        cv2.polylines(out, [p.astype(np.int32)], True, color, 2)
    return cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
