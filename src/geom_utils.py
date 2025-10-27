
# src/geom_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------
# Box conversions
# ----------------------------
def xywh_to_xyxy(box: List[float]) -> List[int]:
    x, y, w, h = box
    return [int(x), int(y), int(x + w), int(y + h)]

def xyxy_to_xywh(box: List[float]) -> List[int]:
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

def clamp_box_xywh(box: List[float], W: int, H: int) -> List[int]:
    x, y, w, h = box
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return [x, y, w, h]

def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

def nms_xywh(boxes: List[List[float]], scores: List[float], iou_thr: float=0.5) -> List[int]:
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou_xywh(boxes[i], boxes[j]) < iou_thr]
    return keep

# ----------------------------
# Letterbox mapping
# ----------------------------
@dataclass
class LetterboxInfo:
    scale_x: float
    scale_y: float
    pad_x: int = 0
    pad_y: int = 0

def map_box_to_original_xywh(box_xywh: List[float], lb: LetterboxInfo) -> List[float]:
    """Map a box from letterboxed/downscaled coords back to original image coords."""
    x, y, w, h = box_xywh
    x0 = (x - lb.pad_x) / lb.scale_x
    y0 = (y - lb.pad_y) / lb.scale_y
    w0 =  w / lb.scale_x
    h0 =  h / lb.scale_y
    return [x0, y0, w0, h0]

def add_offset_xywh(box_xywh: List[float], dx: int, dy: int) -> List[float]:
    x, y, w, h = box_xywh
    return [x + dx, y + dy, w, h]

# ----------------------------
# ROI helpers
# ----------------------------
def pad_box_ratio_xywh(box_xywh: List[float], pad_ratio: float, W: int, H: int) -> List[int]:
    x, y, w, h = box_xywh
    px = int(w * pad_ratio); py = int(h * pad_ratio)
    return clamp_box_xywh([x - px, y - py, w + 2*px, h + 2*py], W, H)

def crop_roi(image, roi_xywh: List[int]):
    """Crop ROI from an OpenCV image and return (crop, (x,y)) offset."""
    x, y, w, h = roi_xywh
    return image[y:y+h, x:x+w].copy(), (x, y)

def remap_tile_box_to_original(tile_box_xywh: List[float], roi_offset: Tuple[int,int], tile_offset: Tuple[int,int]) -> List[int]:
    """tile→ROI→original mapping for (x,y,w,h)."""
    x, y, w, h = tile_box_xywh
    dx_roi, dy_roi = roi_offset
    dx_tile, dy_tile = tile_offset
    return [int(x + dx_roi + dx_tile), int(y + dy_roi + dy_tile), int(w), int(h)]

# ----------------------------
# Tiling
# ----------------------------
def sliding_windows_wh(W, H, tile=1024, overlap=0.2, pad_ratio=0.10):
    """(x1,y1,x2,y2) 리스트 반환. pad_ratio만큼 이미지 바깥으로도 확장 허용."""
    # --- robust casting ---
    W = int(round(W))
    H = int(round(H))
    tile = int(round(tile))
    # overlap은 0~0.9 권장
    overlap = float(overlap)
    pad_ratio = float(pad_ratio)

    # stride 최소 1 보장
    stride = max(1, int(round(tile * (1.0 - overlap))))

    # 이미지가 tile보다 작은 경우도 처리
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))

    pads = []
    pad = int(round(tile * pad_ratio))
    for y in ys:
        for x in xs:
            x1 = x - pad; y1 = y - pad
            x2 = x + tile + pad; y2 = y + tile + pad
            pads.append((x1, y1, x2, y2))

    # 원본 범위로 hard clip
    clipped = []
    for x1, y1, x2, y2 in pads:
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(W, int(x2)), min(H, int(y2))
        if cx2 > cx1 and cy2 > cy1:
            clipped.append((cx1, cy1, cx2, cy2))
    return clipped

