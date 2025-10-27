# src/appDefect.py
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

@dataclass
class DefectRow:
    wheel_id: str
    defect_type: str
    confidence: float
    area_ratio: float
    time: str

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
    return sorted(paths, key=lambda p: os.path.basename(p).lower())

def find_image_by_stem(folder: str, position: str, wheel_id: str) -> str:
    """파일명에 stem(예: '0034')이 포함된 이미지를 찾아 반환"""
    if not folder or not os.path.isdir(folder):
        return ""
    if not wheel_id:
        return ""

    # prefix 생성
    prefix = ""
    pos = position.lower().strip()
    if pos == "right":
        prefix = "r"
    elif pos == "left":
        prefix = "l"

    target = f"{prefix}{wheel_id}".lower()

    for p in list_images(folder):
        name = os.path.splitext(os.path.basename(p))[0].lower()
        if target in name:
            return p
    return ""

def start_run_state(images: List[str]) -> dict:
    order = list(images)
    return {"images_all": images, "order": order, "cursor": 0, "is_running": False}

def advance_state(state: dict, batch_size: int) -> Tuple[List[str], bool, dict]:
    order = state.get("order", [])
    cursor = int(state.get("cursor", 0))
    n = len(order)
    if cursor >= n:
        return [], True, state
    end = min(cursor + batch_size, n)
    batch = order[cursor:end]
    cursor = end
    finished = cursor >= n
    new_state = dict(state)
    new_state["cursor"] = cursor
    return batch, finished, new_state

class Detector:
    def __init__(self, model_path_or_name: str, allowed_defects: List[str], imgsz: int = 768,
                 wheel_classes: List[str] = ("wheel",)):
        self.device = get_device()
        self.model = YOLO(model_path_or_name)
        self.allowed = set(allowed_defects or [])
        self.wheel_classes = set(wheel_classes or [])
        self.imgsz = int(imgsz)

    # ✅ 메모리 기반 워밍업(임시파일 X)
    def warmup(self, conf: float = 0.25):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # BGR
        _ = self.model.predict(
            source=dummy,  # NumPy 배열
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    @staticmethod
    def _area_xyxy(xyxy):
        x1, y1, x2, y2 = xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _inter_area(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return float((ix2 - ix1) * (iy2 - iy1))

    def _post_rows(self, result, conf_thresh: float) -> List[DefectRow]:
        rows: List[DefectRow] = []
        if not hasattr(result, "boxes") or result.boxes is None:
            return rows

        names = result.names or {}
        h, w = (result.orig_img.shape[0], result.orig_img.shape[1]) if hasattr(result, "orig_img") else (1, 1)
        img_area = float(h * w) if h > 0 and w > 0 else 1.0

        try:
            wheel_id = os.path.splitext(os.path.basename(result.path))[0]
        except Exception:
            wheel_id = ""

        # --- 휠 박스 수집 ---
        wheel_boxes_xyxy = []
        for b in result.boxes:
            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            if conf < conf_thresh:
                continue
            cls_idx = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            cls_name = names.get(cls_idx, str(cls_idx))
            if cls_name in self.wheel_classes:
                xyxy = b.xyxy[0].tolist()
                wheel_boxes_xyxy.append(xyxy)

        # --- 기준 휠 선택 (가장 큰 휠) ---
        wheel_xyxy = None
        wheel_area = None
        if wheel_boxes_xyxy:
            wheel_xyxy = max(wheel_boxes_xyxy, key=self._area_xyxy)
            wheel_area = self._area_xyxy(wheel_xyxy)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 결함 누적 ---
        total_defect_area = 0.0
        for b in result.boxes:
            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            if conf < conf_thresh:
                continue
            cls_idx = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            cls_name = names.get(cls_idx, str(cls_idx))
            xyxy = b.xyxy[0].tolist()

            # 휠 기록 (휠 행은 항상 1.0)
            if cls_name in self.wheel_classes:
                rows.append(DefectRow(
                    wheel_id=wheel_id,
                    defect_type=cls_name,
                    confidence=conf,
                    area_ratio=1.0,
                    time=now_str,   # ✅ 현재시간 기록
                ))
                continue

            # 결함 기록 (allowed 필터)
            if self.allowed and cls_name not in self.allowed:
                continue

            if wheel_xyxy is not None and wheel_area and wheel_area > 0:
                inter = self._inter_area(xyxy, wheel_xyxy)
                total_defect_area += inter

            rows.append(DefectRow(
                wheel_id=wheel_id,
                defect_type=cls_name,
                confidence=conf,
                area_ratio=0.0,   # 후에 합산 비율로 업데이트
                time=now_str,     # ✅ 현재시간 기록
            ))

        # --- 전체 결함 비율 계산 (휠 기준 합산) ---
        if wheel_area and wheel_area > 0 and total_defect_area > 0:
            total_ratio = total_defect_area / wheel_area
            for r in rows:
                if r.defect_type not in self.wheel_classes:
                    r.area_ratio = total_ratio

        return rows

    def predict_rows(self, path: str, conf: float) -> List[DefectRow]:
        results = self.model.predict(
            source=path,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._post_rows(results[0], conf)

    def plot_with_boxes(self, path: str, conf: float):
        results = self.model.predict(
            source=path,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return np.zeros((200, 100, 3), dtype=np.uint8)
        rgb = results[0].plot()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)