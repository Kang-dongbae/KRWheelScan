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
# 고정 경로 / 설정
# =======================
# 데이터 루트
DATA_ROOT = Path("/home/dongbae/Dev/WheelScan/data/original_data")
TRAIN_IMAGES = DATA_ROOT / "train/images"
VAL_IMAGES   = DATA_ROOT / "valid/images"
TEST_IMAGES  = DATA_ROOT / "test/images"

# 결과 루트 (WSL UNC 경로)
MODELS_ROOT = Path("/home/dongbae/Dev/WheelScan/models/")
STAGE1_DIR = MODELS_ROOT / "step1"
STAGE2_DIR = MODELS_ROOT / "step2"
STAGE3_DIR = MODELS_ROOT / "step3"
STAGE4_DIR = MODELS_ROOT / "step4"

DATA_YAML = Path("/home/dongbae/Dev/WheelScan/data/original_data/data.yaml")
MODEL_CFG = Path("/home/dongbae/Dev/WheelScan/yolo11m-p2.yaml")
#PRETRAINED_WEIGHTS = "yolov11s.pt"    # 또는 None

TRAIN_CFG = dict(
    
    #save_dir = str(MODELS_ROOT),
    imgsz=1280,          
    epochs=120,         
    batch=2,
    workers=2,            
    seed=42,
    patience=30,        

    box=0.08,            
    cls=0.20,            
    dfl=1.5,            
    #fl_gamma=1.5,       

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

CROP_TRAIN = Path("/home/dongbae/Dev/WheelScan/data/train_tiles")
CROP_VAL   = Path("/home/dongbae/Dev/WheelScan/data/valid_tiles")
CROP_TEST  = Path("/home/dongbae/Dev/WheelScan/data/test_tiles")

# === 타일링 결과를 저장할 루트 (새로 생성됨) ===
TILE_ROOT  = Path("/home/dongbae/Dev/WheelScan/data/tiles_640")
TILE_TRAIN = TILE_ROOT / "train"
TILE_VAL   = TILE_ROOT / "valid"
TILE_TEST  = TILE_ROOT / "test"

# 타일 학습용 data.yaml (결함 2클래스 0/1만 포함)
DATA_YAML_TILES = Path("/home/dongbae/Dev/WheelScan/data/tiles_640/data_tiles.yaml")

# ====== SAHI 설정 ======
SAHI_CFG = dict(
    slice_wh=640,     
    overlap=0.25,      
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


# =======================
# [1단계] 원본 이미지로 학습 (yolo11s-p2)
# =======================
def stage1_train_p2(data_yaml: Path, out_dir: Path) -> Path:
    print("\n=== [1단계] 학습 시작 (yolo11s-p2) ===")
    print(f"data: {data_yaml}")
    print(f"model cfg: {MODEL_CFG}")

    model = YOLO(MODEL_CFG)  # yolo11s-p2.yaml

    # 3) 학습
    train_args = {
        "data": str(data_yaml),
        "project": str(MODELS_ROOT),
        "name": out_dir.name,
        "device": device_str(),

        **TRAIN_CFG,  
    }
    
    results = model.train(**train_args)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[1단계 완료] best weights: {best}")
    return best


# =======================
# [2단계] SAHI 타일 분할
# =======================
def stage2_tile_all_with_sahi(
    tile: int = None,
    overlap: float = None,
    keep_empty: bool = True,
    min_side_px: int = 2,
    min_intersection_ratio: float = 0.2,
):

    t = tile or SAHI_CFG["slice_wh"]
    ov = overlap or SAHI_CFG["overlap"]

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

            # SAHI 슬라이스 창 생성 (추론과 1:1 일치)
            sliced_list = slice_image(
                image=im,
                slice_height=t,
                slice_width=t,
                overlap_height_ratio=ov,
                overlap_width_ratio=ov
            )

            # YOLO 라벨 로드
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

            # 각 슬라이스 저장
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

                    # 타일 좌표계 정규화
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

    print("\n=== [2단계] SAHI로 크롭 세트를 타일 분할 ===")
    tile_one_split(CROP_TRAIN, TILE_TRAIN)
    tile_one_split(CROP_VAL,   TILE_VAL)
    tile_one_split(CROP_TEST,  TILE_TEST)
    print(f"[2단계 완료] 타일 데이터셋 root: {TILE_ROOT} (tile={t}, overlap={ov})")



# =======================
# [3단계] 기본(비-SAHi) 추론
# =======================
def stage3_train_defect_on_tiles(data_yaml_tiles: Path, out_dir: Path) -> Path:
    print("\n=== [3단계] 타일 데이터로 결함 모델 학습 ===")
    print(f"data: {data_yaml_tiles}")
    model = YOLO(MODEL_CFG)
    train_args = {
        "data": str(data_yaml_tiles),
        "project": str(MODELS_ROOT),
        "name": out_dir.name,       # step3
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
# [4단계] SAHI 추론
# =======================
def stage4_infer_on_cropped_with_sahi(weights_path: Path, cropped_test_split: Path, out_dir: Path, sahi_cfg: dict):
    vis_dir = out_dir / "vis"; json_dir = out_dir / "json"
    ensure_dir(vis_dir); ensure_dir(json_dir)

    print("\n=== [4단계] SAHI 추론(크롭 원본 이미지) ===")
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
        res = get_sliced_prediction(
            image=im,
            detection_model=dmodel,
            slice_height=sahi_cfg["slice_wh"],
            slice_width=sahi_cfg["slice_wh"],
            overlap_height_ratio=sahi_cfg["overlap"],
            overlap_width_ratio=sahi_cfg["overlap"],
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
# main (필요 단계 주석 처리해서 실행)
# =======================
def main():
    # --- 1단계: 학습 (yolo11s-p2) ---
    #best_weights = stage1_train_p2(DATA_YAML, STAGE1_DIR)

    # --- 2단계: SAHI 설정 저장 ---
    #stage2_save_sahi_settings(STAGE2_DIR, SAHI_CFG)

    # --- 3단계: 기본(비-SAHi) 추론 ---
    #stage3_predict_baseline(best_weights, TEST_IMAGES, STAGE3_DIR)

    # --- 4단계: SAHI 추론 ---
    #stage4_predict_sahi(best_weights, TEST_IMAGES, STAGE4_DIR, SAHI_CFG)


if __name__ == "__main__":
    # 필요 없는 단계는 아래 호출부를 주석 처리하세요.
    main()
