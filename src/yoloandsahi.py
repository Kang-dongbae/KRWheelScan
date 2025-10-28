#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List

from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import slice_image


# =======================
# 경로 / 설정
# =======================
DATA_ROOT = Path("/home/dongbae/Dev/WheelScan/data/original_data")
TRAIN_IMAGES = DATA_ROOT / "train/images"
VAL_IMAGES   = DATA_ROOT / "valid/images"
TEST_IMAGES  = DATA_ROOT / "test/images"

MODELS_ROOT = Path("/home/dongbae/Dev/WheelScan/models/")
STAGE1_DIR = MODELS_ROOT / "step1"
STAGE2_DIR = MODELS_ROOT / "step2"
STAGE3_DIR = MODELS_ROOT / "step3"
STAGE4_DIR = MODELS_ROOT / "step4"

DATA_YAML = Path("/home/dongbae/Dev/WheelScan/data/original_data/data.yaml")
MODEL_CFG = Path("/home/dongbae/Dev/WheelScan/yolo11m-p2.yaml")

TRAIN_CFG = dict(
    imgsz=640,
    epochs=300,
    batch=4,
    workers=2,
    seed=42,
    patience=30,

    box=0.08,
    cls=0.20,
    dfl=1.5,

    mosaic=0.30,
    copy_paste=0.40,
    mixup=0.0,
    erasing=0.10,
    close_mosaic=10,

    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    translate=0.05,
    scale=0.50,
    hsv_h=0.015, hsv_s=0.50, hsv_v=0.40,
    fliplr=0.2, flipud=0.0,

    rect=True,
    optimizer="AdamW",
    lr0=0.003,
    lrf=0.20,
    weight_decay=0.0005,
    freeze=0,
    amp=True,
    cache=True,
    verbose=False,
    plots=True,
)

# (이미 준비된) 휠-크롭 데이터셋
CROP_TRAIN = Path("/home/dongbae/Dev/WheelScan/data/train_tiles")
CROP_VAL   = Path("/home/dongbae/Dev/WheelScan/data/valid_tiles")
CROP_TEST  = Path("/home/dongbae/Dev/WheelScan/data/test_tiles")

# 타일링 결과 저장 루트
TILE_ROOT  = Path("/home/dongbae/Dev/WheelScan/data/tiles_out")
TILE_TRAIN = TILE_ROOT / "train"
TILE_VAL   = TILE_ROOT / "valid"
TILE_TEST  = TILE_ROOT / "test"

# 타일 학습용 data.yaml (수동 생성)
DATA_YAML_TILES = TILE_ROOT / "data_tiles.yaml"

# ====== SAHI 설정 (스위치 가능) ======
SAHI_CFG = dict(
    # --- 분할 방식 ---
    # "size"   : 고정 크기 타일 (SPLIT_VALUE=타일 변 px)
    # "count_v": 세로 N등분 (SPLIT_VALUE=N)
    SPLIT_FLAG="count_v",
    SPLIT_VALUE=6,

    # --- 겹침 비율 ---
    overlap_h=0.10,   # 세로 겹침
    overlap_w=0.00,   # 가로 겹침 (count_v 모드면 보통 0.0 권장)

    # --- 추론/후처리 ---
    conf_thres=0.18,
    postprocess="NMS",
    match_metric="IOU",
    match_thres=0.45
)


# =======================
# 유틸
# =======================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(dir_path: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files: List[Path] = []
    for e in exts:
        files.extend(dir_path.glob(e))
    return sorted(files)

def device_str() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def compute_slice_params(W: int, H: int, cfg: dict):
    """SPLIT_FLAG/SPLIT_VALUE에 따라 slice_height/width와 overlap 비율을 계산"""
    flag = cfg.get("SPLIT_FLAG", "size")
    val  = int(cfg.get("SPLIT_VALUE", 640))
    ovh  = float(cfg.get("overlap_h", 0.25))
    ovw  = float(cfg.get("overlap_w", 0.25))

    if flag == "count_v":  # 세로 등분
        slice_h = max(1, H // max(1, val))
        slice_w = W
        return slice_h, slice_w, ovh, 0.0
    else:                  # "size": 정사각 타일
        slice_h = val
        slice_w = val
        return slice_h, slice_w, ovh, ovw


# =======================
# [1단계] (옵션) 원본 학습
# =======================
def stage1_train_p2(data_yaml: Path, out_dir: Path) -> Path:
    print("\n=== [1단계] 학습 시작 (yolo11m-p2) ===")
    print(f"data: {data_yaml}")
    print(f"model cfg: {MODEL_CFG}")

    model = YOLO(MODEL_CFG)
    train_args = {
        "data": str(data_yaml),
        "project": str(MODELS_ROOT),
        "name": out_dir.name,
        "device": device_str(),
        **TRAIN_CFG,
        "exist_ok": True,
    }
    results = model.train(**train_args)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[1단계 완료] best weights: {best}")
    return best


# =======================
# [2단계] SAHI 타일 분할 (size/count_v 스위치)
# =======================
def stage2_tile_all_with_sahi(
    keep_empty: bool = True,
    min_side_px: int = 2,
    min_intersection_ratio: float = 0.2,
):
    """
    CROP_TRAIN/CROP_VAL/CROP_TEST (images/labels)
      → SAHI 규칙으로 타일 분할 → TILE_TRAIN/TILE_VAL/TILE_TEST 저장
    """
    def tile_one_split(src_split: Path, dst_split: Path):
        src_img = src_split / "images"
        src_lbl = src_split / "labels"
        dst_img = dst_split / "images"
        dst_lbl = dst_split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        for ip in sorted(src_img.iterdir()):
            if ip.suffix.lower() not in exts:
                continue

            im = Image.open(ip).convert("RGB")
            W, H = im.size
            slice_h, slice_w, ovh, ovw = compute_slice_params(W, H, SAHI_CFG)

            sliced_list = slice_image(
                image=im,
                slice_height=slice_h,
                slice_width=slice_w,
                overlap_height_ratio=ovh,
                overlap_width_ratio=ovw
            )

            # 원본 YOLO 라벨(px 좌표로 변환)
            ypath = src_lbl / (ip.stem + ".txt")
            boxes = []
            if ypath.exists():
                with open(ypath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls = int(parts[0])
                        cx, cy, ww, hh = map(float, parts[1:5])
                        bx = cx*W; by = cy*H; bw = ww*W; bh = hh*H
                        x1 = bx - bw/2; y1 = by - bh/2
                        x2 = bx + bw/2; y2 = by + bh/2
                        boxes.append((cls, x1,y1,x2,y2, bw*bh))  # (cls, x1,y1,x2,y2, area)

            # 각 슬라이스 저장 + 라벨 클리핑/정규화
            for si in sliced_list:
                x0, y0 = si.start_x, si.start_y
                x1t, y1t = si.end_x, si.end_y
                crop = im.crop((x0, y0, x1t, y1t))
                tw, th = crop.size

                new_lines = []
                for (cls, x1,y1,x2,y2, area) in boxes:
                    ix1 = max(x1, x0); iy1 = max(y1, y0)
                    ix2 = min(x2, x1t); iy2 = min(y2, y1t)
                    iw = ix2 - ix1; ih = iy2 - iy1
                    if iw <= 0 or ih <= 0:
                        continue
                    if iw < min_side_px or ih < min_side_px:
                        continue
                    if (iw*ih) / (area + 1e-6) < min_intersection_ratio:
                        continue

                    cx_t = ((ix1 + ix2)/2 - x0) / tw
                    cy_t = ((iy1 + iy2)/2 - y0) / th
                    w_t  = iw / tw
                    h_t  = ih / th

                    if w_t <= 0 or h_t <= 0:
                        continue
                    cx_t = min(max(cx_t, 0.0), 1.0)
                    cy_t = min(max(cy_t, 0.0), 1.0)
                    w_t  = min(max(w_t, 1e-6), 1.0)
                    h_t  = min(max(h_t, 1e-6), 1.0)

                    new_lines.append(f"{cls} {cx_t:.6f} {cy_t:.6f} {w_t:.6f} {h_t:.6f}")

                tile_name = f"{ip.stem}_{x0}_{y0}"
                crop.save(dst_img / f"{tile_name}.jpg", quality=95)
                with open(dst_lbl / f"{tile_name}.txt", "w") as f:
                    if new_lines or keep_empty:
                        f.write("\n".join(new_lines))

    print("\n=== [2단계] SAHI 타일 분할 시작 ===")
    tile_one_split(CROP_TRAIN, TILE_TRAIN)
    tile_one_split(CROP_VAL,   TILE_VAL)
    tile_one_split(CROP_TEST,  TILE_TEST)
    print(f"[2단계 완료] 타일 데이터셋 root: {TILE_ROOT}")
    print(f" - 모드: {SAHI_CFG['SPLIT_FLAG']}, 값: {SAHI_CFG['SPLIT_VALUE']}")
    print(f" - overlap_h: {SAHI_CFG['overlap_h']}, overlap_w: {SAHI_CFG['overlap_w']}")


# =======================
# [3단계] 타일 데이터 학습
# =======================
def stage3_train_defect_on_tiles(data_yaml_tiles: Path, out_dir: Path) -> Path:
    print("\n=== [3단계] 타일 데이터로 결함 모델 학습 ===")
    print(f"data: {data_yaml_tiles}")
    model = YOLO(MODEL_CFG)
    train_args = {
        "data": str(data_yaml_tiles),
        "project": str(MODELS_ROOT),
        "name": out_dir.name,
        "device": device_str(),
        **TRAIN_CFG,
        "plots": True,
        "exist_ok": True,
    }
    results = model.train(**train_args)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[3단계 완료] best weights: {best}")
    return best


# =======================
# [4단계] SAHI 추론 (타일 규칙 동일)
# =======================
def stage4_infer_on_cropped_with_sahi(weights_path: Path, cropped_test_split: Path, out_dir: Path, sahi_cfg: dict):
    vis_dir = out_dir / "vis"; json_dir = out_dir / "json"
    ensure_dir(vis_dir); ensure_dir(json_dir)

    print("\n=== [4단계] SAHI 추론 시작 ===")
    dmodel = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=sahi_cfg["conf_thres"],
        device=device_str()
    )

    imgs = list_images(cropped_test_split / "images")
    if not imgs:
        raise FileNotFoundError(f"크롭 테스트 이미지가 없습니다: {cropped_test_split/'images'}")

    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        slice_h, slice_w, ovh, ovw = compute_slice_params(W, H, sahi_cfg)

        res = get_sliced_prediction(
            image=im,
            detection_model=dmodel,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=ovh,
            overlap_width_ratio=ovw,
            postprocess_type=sahi_cfg["postprocess"],
            postprocess_match_metric=sahi_cfg["match_metric"],
            postprocess_match_threshold=sahi_cfg["match_thres"],
            postprocess_class_agnostic=True
        )

        stem = Path(ip).stem
        res.export_visualization(export_dir=str(vis_dir), file_name=f"{stem}_sahi_vis.jpg")
        res.to_coco_annotations(save_path=str(json_dir / f"{stem}_pred.json"))

    print(f"[4단계 완료] 시각화: {vis_dir}")
    print(f"[4단계 완료] COCO preds: {json_dir}")


# =======================
# main
# =======================
def main():
    # 1단계 (옵션): 이미 원본 학습 끝났으면 생략
    # best_wheel = stage1_train_p2(DATA_YAML, STAGE1_DIR)

    # 2단계: SAHI 타일 분할 (size/count_v 모드 중 택1)
    stage2_tile_all_with_sahi()

    # 3단계: 타일 학습
    #best_defect = stage3_train_defect_on_tiles(DATA_YAML_TILES, STAGE3_DIR)

    # 4단계: SAHI 추론 (2단계와 동일 규칙)
    #stage4_infer_on_cropped_with_sahi(best_defect, CROP_TEST, STAGE4_DIR, SAHI_CFG)


if __name__ == "__main__":
    main()
