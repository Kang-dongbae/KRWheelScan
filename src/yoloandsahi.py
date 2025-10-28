#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List

from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


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
MODEL_CFG = Path("/home/dongbae/Dev/WheelScan/yolo11s-p2.yaml")
#PRETRAINED_WEIGHTS = "yolov11s.pt"    # 또는 None

TRAIN_CFG = dict(
    
    #save_dir = str(MODELS_ROOT),
    imgsz=1280,          
    epochs=120,         
    batch=6,            
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
    workers=4,
    cache=True,
    verbose=False,
    plots=False,
)

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
# [2단계] SAHI 설정 저장(간단)
# =======================
def stage2_save_sahi_settings(out_dir: Path, sahi_cfg: dict):
    cfg_path = out_dir / "sahi_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(sahi_cfg, f, indent=2, ensure_ascii=False)
    print("\n=== [2단계] SAHI 설정 저장 ===")
    print(f"저장: {cfg_path}")


# =======================
# [3단계] 기본(비-SAHi) 추론
# =======================
def stage3_predict_baseline(weights_path: Path, test_images: Path, out_dir: Path):
    print("\n=== [3단계] 기본(비-SAHi) 추론 시작 ===")
    model = YOLO(str(weights_path))
    model.predict(
        source=str(test_images),
        save=True,
        project=str(MODELS_ROOT),
        name=out_dir.name,   # stage3
        conf=0.25,
        iou=0.50,
        imgsz=TRAIN_CFG["imgsz"],
        device=device_str(),
    )
    print(f"[3단계 완료] 결과 저장: {out_dir}")


# =======================
# [4단계] SAHI 추론
# =======================
def stage4_predict_sahi(weights_path: Path, test_images: Path, out_dir: Path, sahi_cfg: dict):
    vis_dir = out_dir / "vis"
    json_dir = out_dir / "json"
    ensure_dir(vis_dir); ensure_dir(json_dir)

    print("\n=== [4단계] SAHI(슬라이스&타일) 추론 시작 ===")
    dmodel = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=sahi_cfg["conf_thres"],
        device=device_str()
    )
    imgs = list_images(test_images)
    if not imgs:
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {test_images}")

    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        res = get_sliced_prediction(
            image=im,
            detection_model=dmodel,
            slice_height=sahi_cfg["slice_wh"],
            slice_width=sahi_cfg["slice_wh"],
            overlap_height_ratio=sahi_cfg["overlap"],
            overlap_width_ratio=sahi_cfg["overlap"],
            postprocess_type=sahi_cfg["postprocess"],           # "NMS" or "GREEDYNMM"
            postprocess_match_metric=sahi_cfg["match_metric"],  # "IOU"
            postprocess_match_threshold=sahi_cfg["match_thres"],
            postprocess_class_agnostic=True
        )
        stem = Path(ip).stem
        # 시각화
        res.export_visualization(
            export_dir=str(vis_dir),
            file_name=f"{stem}_sahi_vis.jpg"
        )
        # per-image COCO 예측 저장
        res.to_coco_annotations(save_path=str(json_dir / f"{stem}_pred.json"))

    print(f"[4단계 완료] 시각화: {vis_dir}")
    print(f"[4단계 완료] per-image COCO preds: {json_dir}")


# =======================
# main (필요 단계 주석 처리해서 실행)
# =======================
def main():
    # --- 1단계: 학습 (yolo11s-p2) ---
    best_weights = stage1_train_p2(DATA_YAML, STAGE1_DIR)

    # --- 2단계: SAHI 설정 저장 ---
    #stage2_save_sahi_settings(STAGE2_DIR, SAHI_CFG)

    # --- 3단계: 기본(비-SAHi) 추론 ---
    #stage3_predict_baseline(best_weights, TEST_IMAGES, STAGE3_DIR)

    # --- 4단계: SAHI 추론 ---
    #stage4_predict_sahi(best_weights, TEST_IMAGES, STAGE4_DIR, SAHI_CFG)


if __name__ == "__main__":
    # 필요 없는 단계는 아래 호출부를 주석 처리하세요.
    main()
