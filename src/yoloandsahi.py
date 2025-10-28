#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List

from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.slicing import slice_image

import random
import shutil
from collections import defaultdict

#from sahi.predict import get_sliced_prediction, save_sliced_prediction_json
#from sahi.predict import get_sliced_prediction
#from sahi.utils.file import export_visualizations
#from sahi.utils.file import export_visualizations
#from sahi.utils.file import export_visualizations
import sahi # sahi 패키지 전체를 임포트합니다.

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
    batch=6,
    workers=4,
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
FINAL_ROOT = Path("/home/dongbae/Dev/WheelScan/data/final_splits")
TILE_TRAIN = TILE_ROOT / "train"
TILE_VAL   = TILE_ROOT / "valid"
TILE_TEST  = TILE_ROOT / "test"

# 타일 학습용 data.yaml (수동 생성)
DATA_YAML_TILES = FINAL_ROOT / "data_tiles.yaml"

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
    conf_thres=0.8,
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
from pathlib import Path
from PIL import Image
# compute_slice_params, slice_image, CROP_TRAIN, CROP_VAL, CROP_TEST, TILE_ROOT, SAHI_CFG 등은 
# 외부에서 정의된 것으로 가정하고 코드를 작성합니다.

def stage2_tile_all_with_sahi(
    keep_empty: bool = True,
    min_side_px: int = 2,
    min_intersection_ratio: float = 0.2,
):
    """
    CROP_TRAIN/CROP_VAL/CROP_TEST (images/labels)의 모든 이미지를 타일 분할하여
    단일 폴더 (TILE_ROOT)에 저장합니다.
    """
    
    # TILE_ROOT가 /home/dongbae/Dev/WheelScan/data/tiles_out 경로라고 가정
    # 이 폴더 아래 images와 labels를 만들고 모든 타일 결과를 저장합니다.
    dst_img_root = TILE_ROOT / "images"
    dst_lbl_root = TILE_ROOT / "labels"
    dst_img_root.mkdir(parents=True, exist_ok=True)
    dst_lbl_root.mkdir(parents=True, exist_ok=True)

    def tile_one_split(src_split: Path):
        """
        특정 분할(src_split)의 모든 이미지를 가져와 TILE_ROOT에 타일링 결과를 저장합니다.
        """
        src_img = src_split / "images"
        src_lbl = src_split / "labels"

        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        for ip in sorted(src_img.iterdir()):
            if ip.suffix.lower() not in exts:
                continue

            im = Image.open(ip).convert("RGB")
            W, H = im.size
            
            # SAHI 파라미터 계산은 한 번만 수행
            # 이 파라미터는 SAHI_CFG에 따라 정해지며, 모든 이미지에 동일하게 적용됩니다.
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
                
                # --- SAHI 좌표 추출 및 크기 계산 (이전 수정 반영) ---
                x0, y0 = si["starting_pixel"]
                # tw_slice, th_slice = si["image"].size 로 인한 에러 대신 slice_w, slice_h 사용
                tw_slice, th_slice = slice_w, slice_h 

                x1t, y1t = x0 + tw_slice, y0 + th_slice
                
                # 원본 이미지에서 타일 자르기
                crop = im.crop((x0, y0, x1t, y1t))
                tw, th = crop.size # 타일의 실제 크기 (tw, th)를 사용하여 정규화
                # ------------------------------------------------

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
                    w_t = min(max(w_t, 1e-6), 1.0) # 0 또는 1.0 초과 방지
                    h_t  = ih / th
                    h_t = min(max(h_t, 1e-6), 1.0) # 0 또는 1.0 초과 방지


                    if w_t <= 0 or h_t <= 0:
                        continue
                    cx_t = min(max(cx_t, 0.0), 1.0)
                    cy_t = min(max(cy_t, 0.0), 1.0)

                    new_lines.append(f"{cls} {cx_t:.6f} {cy_t:.6f} {w_t:.6f} {h_t:.6f}")

                # 타일 이름을 '원본이미지명_x시작좌표_y시작좌표'로 통일하여 단일 폴더에 저장
                tile_name = f"{ip.stem}_{x0}_{y0}"
                crop.save(dst_img_root / f"{tile_name}.jpg", quality=95)
                with open(dst_lbl_root / f"{tile_name}.txt", "w") as f:
                    if new_lines or keep_empty:
                        f.write("\n".join(new_lines))

    print("\n=== [2단계] SAHI 타일 분할 시작 (단일 출력 모드) ===")
    
    # TILE_ROOT 폴더에 모든 결과를 저장하기 위해 train, val, test를 순차적으로 처리
    tile_one_split(CROP_TRAIN)
    tile_one_split(CROP_VAL)
    tile_one_split(CROP_TEST)
    
    print(f"[2단계 완료] 타일 데이터셋 root: {TILE_ROOT}")
    print(f" - 출력 폴더: {dst_img_root.parent}")
    print(f" - 모드: {SAHI_CFG['SPLIT_FLAG']}, 값: {SAHI_CFG['SPLIT_VALUE']}")
    print(f" - overlap_h: {SAHI_CFG['overlap_h']}, overlap_w: {SAHI_CFG['overlap_w']}")


#===============================
# 2.5단계 : 데이터 오버샘플링
#===============================

def create_iterative_splits(tile_root: Path, num_iterations: int = 8, train_ratio: float = 0.8):
    
    label_dir = tile_root / "labels"
    image_dir = tile_root / "images"
    
    # 최종 출력 디렉토리 설정
    final_output_root = tile_root.parent / "final_splits"

    # 이전 결과 삭제 및 새 폴더 생성
    if final_output_root.exists():
        shutil.rmtree(final_output_root)
    
    # 1. 모든 라벨 파일 분류 (이전과 동일)
    all_label_files = list(label_dir.glob("*.txt"))
    
    class_0_files = [] 
    class_1_files = [] 
    empty_files = []   

    for label_path in all_label_files:
        content = label_path.read_text().strip()
        if not content:
            empty_files.append(label_path)
            continue
        
        classes = {int(line.split()[0]) for line in content.split('\n') if line}
        has_class_1 = 1 in classes
        has_class_0 = 0 in classes

        if has_class_1:
             class_1_files.append(label_path)
        elif has_class_0:
            class_0_files.append(label_path)
        else:
            class_0_files.append(label_path) 
    
    # 2. 반복 추출을 위한 변수 초기화
    # 최종적으로 복제될 파일의 목록. (original_path, new_name_stem) 튜플 저장
    final_replicated_list = [] 
    
    # 정상 파일 처리를 위한 변수 초기화 (비복원 추출, 1/8씩)
    random.shuffle(empty_files)
    num_empty_files = len(empty_files)
    empty_split_size = num_empty_files // num_iterations 
    empty_current_index = 0
    
    # 결함 1번 처리를 위한 변수 초기화 (혼합 추출)
    num_class_1 = len(class_1_files)
    class_1_total_to_extract = num_class_1 // 2 
    class_1_extract_cycle_count = num_iterations // 2 # 총 4번의 사이클
    
    class_1_files_for_loop = list(class_1_files)
    class_1_current_index = 0
    
    
    # 3. 8번 반복하면서 복제 리스트 구성
    for i in range(num_iterations):
        
        # 3-1. 결함 0번 파일 (모든 반복마다 복제: 총 8번 복제)
        for label_path in class_0_files:
            new_name = f"{label_path.stem}_{i}"
            final_replicated_list.append((label_path, new_name))
        
        # 3-2. 결함 1번 파일 (2루프마다 1/2씩 비복원 추출, 총 4번 복제)
        
        # 새로운 4사이클의 시작 (i=0, 2, 4, 6)
        if i % 2 == 0: 
            # 1. 전체 파일 셔플 (복원 추출 효과)
            random.shuffle(class_1_files_for_loop)
            # 2. 인덱스 리셋 (비복원 추출 시작)
            class_1_current_index = 0
            
            # 2번의 루프에 걸쳐 추출될 양 (1/2)을 2로 나눔 (1/4)
            class_1_split_size = class_1_total_to_extract // 2
            
        # 추출 실행
        if class_1_current_index < num_class_1: 
            start_index = class_1_current_index
            
            # 짝수 루프(i=0, 2, 4, 6)에서는 1/4 추출
            if i % 2 == 0: 
                end_index = min(start_index + class_1_split_size, num_class_1)
            # 홀수 루프(i=1, 3, 5, 7)에서는 나머지 1/4 추출
            else: 
                # 남은 1/2 중 나머지를 모두 추출
                end_index = start_index + (class_1_total_to_extract - class_1_split_size)
                end_index = min(end_index, num_class_1) 
                
            selected_class_1 = class_1_files_for_loop[start_index:end_index]
            
            # 추출된 파일은 복제됩니다. (총 4번 복제 로직)
            # 복제 인덱스: 0, 0, 1, 1, 2, 2, 3, 3 -> i // 2 사용
            replicate_idx = i // 2
            for label_path in selected_class_1:
                new_name = f"{label_path.stem}_{replicate_idx}"
                final_replicated_list.append((label_path, new_name))
                
            class_1_current_index = end_index # 인덱스 업데이트 (비복원)
            
        # 3-3. 빈 파일 (정상) (1/8 비복원 추출: 1번만 추가)
        start_index = empty_current_index
        end_index = min(empty_current_index + empty_split_size, num_empty_files)
        
        if i == num_iterations - 1:
            end_index = num_empty_files
            
        selected_empty = empty_files[start_index:end_index]
        
        # 빈 파일은 복제하지 않고, 원본 파일명 그대로 final_replicated_list에 추가
        for label_path in selected_empty:
            final_replicated_list.append((label_path, label_path.stem))
            
        empty_current_index = end_index

    # 4. 최종 데이터셋 분할 (8:2)
    # final_replicated_list: 복제되어 중복된 파일(Path, name)의 총 목록
    random.shuffle(final_replicated_list)
    
    num_total = len(final_replicated_list)
    num_train = int(num_total * train_ratio)
    
    train_replicated = final_replicated_list[:num_train]
    valid_replicated = final_replicated_list[num_train:]

    print(f"\n--- 최종 Train/Valid 분할 (총 {num_total}개 파일) ---")
    print(f"Train 셋 (복제): {len(train_replicated)}개 파일 ({train_ratio*100:.0f}%)")
    print(f"Valid 셋 (복제): {len(valid_replicated)}개 파일 ({(1-train_ratio)*100:.0f}%)")

    # 5. 파일 복사 및 데이터셋 생성
    train_output_dir = final_output_root / "train"
    valid_output_dir = final_output_root / "valid"
    (train_output_dir / "images").mkdir(parents=True)
    (train_output_dir / "labels").mkdir(parents=True)
    (valid_output_dir / "images").mkdir(parents=True)
    (valid_output_dir / "labels").mkdir(parents=True)
    
    def replicate_and_copy(replicated_list, target_dir):
        """원본 파일을 읽어와 새로운 이름으로 이미지와 라벨을 복제 및 저장합니다."""
        target_img_dir = target_dir / "images"
        target_lbl_dir = target_dir / "labels"
        
        for original_path, new_stem in replicated_list:
            original_stem = original_path.stem
            
            # 이미지 파일 경로 (모든 타일 이미지를 .jpg로 가정)
            original_image_path = image_dir / f"{original_stem}.jpg"
            
            if original_image_path.exists() and original_path.exists():
                # 새 파일명
                new_image_name = f"{new_stem}.jpg"
                new_label_name = f"{new_stem}.txt"
                
                # 복제 및 저장
                shutil.copy(original_image_path, target_img_dir / new_image_name)
                shutil.copy(original_path, target_lbl_dir / new_label_name)
                
    replicate_and_copy(train_replicated, train_output_dir)
    replicate_and_copy(valid_replicated, valid_output_dir)

    print(f"\n✅ 데이터셋 생성 완료! 출력 경로: {final_output_root}")
    print(f"   - Train/Valid images/labels에 복제 파일 저장 완료")

    return final_output_root


# =======================
# [2.6단계] 오버샘플링 - 결함데이터만
# =======================

# 전역 경로 설정을 함수 내부에서 사용한다고 가정합니다.
from typing import List, Tuple
def oversample_tiles_for_2_loops(tile_root: Path, train_ratio: float = 0.8) -> Path:
    """
    타일 데이터셋(TILE_ROOT)에서 결함 데이터를 2번의 루프를 통해 오버샘플링하고,
    최종적으로 훈련/검증(Train/Valid) 세트로 분할하여 FINAL_ROOT에 저장합니다.
    (주의: 라벨이 비어있는 파일은 최종 데이터셋에서 완전히 제외됩니다.)

    Args:
        tile_root: 타일 이미지와 라벨이 저장된 루트 경로 (예: /data/tiles_out)
        final_root: 최종 분할 데이터를 저장할 경로 (예: /data/final_splits)
        train_ratio: 훈련 세트 비율 (기본 0.8)

    Returns:
        최종 데이터셋이 저장된 경로 (Path)
    """
    label_dir = tile_root / "labels"
    image_dir = tile_root / "images"
    num_loops = 2

    final_root = tile_root.parent / "final_splits"

    # 1. 이전 결과 삭제 및 새 폴더 생성
    if final_root.exists():
        shutil.rmtree(final_root)

    # 2. 모든 라벨 파일 분류
    all_label_files: List[Path] = list(label_dir.glob("*.txt"))

    class_0_files: List[Path] = []  # 결함0 (복제 대상)
    class_1_files: List[Path] = []  # 결함1 (나눠서 추출 대상)
    # empty_files는 제외하므로 리스트를 생성하지 않습니다.

    for label_path in all_label_files:
        content = label_path.read_text().strip()
        if not content:
            # ⭐ 라벨이 비어있는 파일(정상/배경)은 리스트에 추가하지 않고 건너뜁니다.
            continue 

        classes = {int(line.split()[0]) for line in content.split('\n') if line}
        has_class_1 = 1 in classes
        has_class_0 = 0 in classes

        if has_class_1:
            class_1_files.append(label_path)
        elif has_class_0:
            class_0_files.append(label_path)
        
        # class 0, 1 외의 클래스는 무시하거나, 필요에 따라 처리 로직을 추가할 수 있습니다.

    # 3. 루프를 돌며 복제 리스트 구성
    final_replicated_list: List[Tuple[Path, str]] = []  # (original_path, new_name_stem)

    # 결함1 데이터를 절반씩 나누기 위해 섞습니다.
    random.shuffle(class_1_files)
    num_class_1 = len(class_1_files)
    half_class_1 = num_class_1 // 2
    
    for i in range(num_loops):
        print(f"--- 오버샘플링 루프 {i+1}/{num_loops} ---")
        
        # 3-1. 그룹0 (결함0) 파일 복제/추출 (모든 루프에서 복제)
        for label_path in class_0_files:
            # 복제 파일을 원본과 구별하기 위해 루프 인덱스(i)를 파일명에 추가
            new_name = f"{label_path.stem}_d0_{i+1}" 
            final_replicated_list.append((label_path, new_name))
        
        # 3-2. 그룹1 (결함1) 파일 절반 추출 (비복원 추출)
        if i == 0:
            selected_class_1 = class_1_files[:half_class_1]
        else: # i == 1
            selected_class_1 = class_1_files[half_class_1:]

        # 결함1 파일은 루프 인덱스를 붙여 최종 데이터셋에 서로 다른 파일로 존재하게 합니다.
        for label_path in selected_class_1:
            new_name = f"{label_path.stem}_d1_{i+1}"
            final_replicated_list.append((label_path, new_name))
            
        # ❌ 3-3. 정상(배경) 파일 추출 로직은 완전히 제거되었습니다.

    # 4. 최종 데이터셋 분할 (8:2)
    random.shuffle(final_replicated_list)
    
    num_total = len(final_replicated_list)
    num_train = int(num_total * train_ratio)
    
    train_replicated = final_replicated_list[:num_train]
    valid_replicated = final_replicated_list[num_train:]

    print(f"\n--- 최종 Train/Valid 분할 (총 {num_total}개 파일) ---")
    print(f"Train 셋 (복제): {len(train_replicated)}개 파일 ({train_ratio*100:.0f}%)")
    print(f"Valid 셋 (복제): {len(valid_replicated)}개 파일 ({(1-train_ratio)*100:.0f}%)")

    # 5. 파일 복사 및 데이터셋 생성
    train_output_dir = final_root / "train"
    valid_output_dir = final_root / "valid"
    (train_output_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (valid_output_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    def replicate_and_copy(replicated_list, target_dir):
        target_img_dir = target_dir / "images"
        target_lbl_dir = target_dir / "labels"
        
        for original_path, new_stem in replicated_list:
            original_stem = original_path.stem
            original_image_path = image_dir / f"{original_stem}.jpg"
            
            if original_image_path.exists() and original_path.exists():
                new_image_name = f"{new_stem}.jpg"
                new_label_name = f"{new_stem}.txt"
                
                shutil.copy(original_image_path, target_img_dir / new_image_name)
                shutil.copy(original_path, target_lbl_dir / new_label_name)
                
    replicate_and_copy(train_replicated, train_output_dir)
    replicate_and_copy(valid_replicated, valid_output_dir)

    print(f"\n✅ 데이터셋 생성 완료! 출력 경로: {final_root}")

    return final_root

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
def _measure_text(draw, text: str, font=None):
    # Pillow >= 8.0
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # Fallbacks
    if font is not None and hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    if font is not None and hasattr(font, "getsize"):
        return font.getsize(text)
    # 아주 보수적인 최후의 추정
    return (max(1, int(0.6 * 12 * len(text))), 12)

from PIL import Image, ImageDraw, ImageFont
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def stage4_infer_yolo_with_sahi(
    weights_path: Path,
    cropped_test_split: Path,
    out_dir: Path,
    sahi_cfg: dict,
    keep_empty: bool = True,
    save_vis: bool = True,
):
    """
    테스트(크롭 휠) 이미지를 SAHI 슬라이싱으로 추론하고,
    결과를 YOLO 포맷(txt, 'cls cx cy w h conf')으로 저장합니다.
    - 슬라이싱 규칙: sahi_cfg (훈련과 동일)
    - postprocess: sahi_cfg (NMS/IOU/threshold 등)
    """
    lbl_dir = out_dir / "labels"
    vis_dir = out_dir / "images_vis"
    ensure_dir(lbl_dir)
    if save_vis:
        ensure_dir(vis_dir)

    print("\n=== [4단계] SAHI 추론 (YOLO 포맷 저장) ===")
    dmodel = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=sahi_cfg.get("conf_thres", 0.5),
        device=device_str(),
    )

    imgs = list_images(cropped_test_split / "images")
    if not imgs:
        raise FileNotFoundError(f"크롭 테스트 이미지가 없습니다: {cropped_test_split/'images'}")

    # 폰트는 선택(없는 환경 고려)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None

    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        slice_h, slice_w, ovh, ovw = compute_slice_params(W, H, sahi_cfg)

        # SAHI 슬라이스 추론 (+ 병합 후처리)
        res = get_sliced_prediction(
            image=im,
            detection_model=dmodel,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=ovh,
            overlap_width_ratio=ovw,
            postprocess_type=sahi_cfg.get("postprocess", "NMS"),
            postprocess_match_metric=sahi_cfg.get("match_metric", "IOU"),
            postprocess_match_threshold=sahi_cfg.get("match_thres", 0.45),
            postprocess_class_agnostic=True,
        )

        # 🔥 confidence filtering 추가
       
        stem = Path(ip).stem
        yolo_lines = []

        # SAHI는 중복을 병합한 object_prediction_list를 제공
        for op in res.object_prediction_list:
            # bbox: VOC(xmin, ymin, xmax, ymax)
            x1, y1, x2, y2 = map(float, op.bbox.to_voc_bbox())
            # YOLO 정규화(cx, cy, w, h)
            bw = max(1e-6, x2 - x1)
            bh = max(1e-6, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cxn = min(max(cx / W, 0.0), 1.0)
            cyn = min(max(cy / H, 0.0), 1.0)
            bwn = min(max(bw / W, 1e-6), 1.0)
            bhn = min(max(bh / H, 1e-6), 1.0)

            # 클래스/점수
            cls_id = getattr(op.category, "id", 0)
            try:
                cls_id = int(cls_id)
            except Exception:
                cls_id = 0
            score = getattr(op.score, "value", None)
            conf = 0.0 if score is None else float(score)

            yolo_lines.append(f"{cls_id} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f} {conf:.4f}")

        # 신뢰도 높은 순 정렬(선택 사항)
        if yolo_lines:
            yolo_lines = sorted(
                yolo_lines,
                key=lambda s: float(s.strip().split()[-1]),
                reverse=True,
            )

        # YOLO 라벨 저장
        out_txt = lbl_dir / f"{stem}.txt"
        if yolo_lines or keep_empty:
            with open(out_txt, "w") as f:
                f.write("\n".join(yolo_lines))
        else:
            # keep_empty=False 이고 예측 없으면 파일 미생성
            pass

        # 간단 시각화(선택)
        if save_vis:
            try:
                vis = im.copy()
                draw = ImageDraw.Draw(vis)
                for op in res.object_prediction_list:
                    x1, y1, x2, y2 = map(int, op.bbox.to_voc_bbox())
                    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

                    cls_name = getattr(op.category, "name", None)
                    cls_id = getattr(op.category, "id", None)
                    label = str(cls_name if cls_name is not None else cls_id)
                    score = getattr(op.score, "value", None)
                    if score is not None:
                        label = f"{label}:{score:.2f}"

                    if label:
                        tw, th = _measure_text(draw, label, font=font)
                        tx, ty = x1, max(0, y1 - th - 2)
                        # 텍스트 배경 박스
                        draw.rectangle([(tx, ty), (tx + tw + 2, ty + th + 2)], fill=(255, 0, 0))
                        draw.text((tx + 1, ty + 1), label, fill=(255, 255, 255), font=font)

                vis.save((vis_dir / f"{stem}.png"))
            except Exception as e:
                print(f"⚠️ {ip.name} 시각화 저장 중 오류: {e}")

    print(f"[4단계 완료] YOLO labels: {lbl_dir}")
    if save_vis:
        print(f"[4단계 완료] 시각화: {vis_dir}")


# =======================
# main
# =======================
def main():
    # 1단계 (옵션): 이미 원본 학습 끝났으면 생략
    # best_wheel = stage1_train_p2(DATA_YAML, STAGE1_DIR)

    # 2단계: SAHI 타일 분할 (size/count_v 모드 중 택1)
    #stage2_tile_all_with_sahi()
    
    # 2.5단계 : 데이터 오버샘플링
    #final_output_path = create_iterative_splits(tile_root=TILE_ROOT)
    #print(f"\n✨ 최종 Train/Valid 데이터셋 생성 완료 위치: {final_output_path}")
    
    # 2.6단계 : 데이터 오버샘플링 - 결함데이터만
    #final_output_path = oversample_tiles_for_2_loops(tile_root=TILE_ROOT)
    #print(f"\n✨ 최종 Train/Valid 데이터셋 생성 완료 위치: {final_output_path}")


    # 3단계: 타일 학습
    #best_defect = stage3_train_defect_on_tiles(DATA_YAML_TILES, STAGE3_DIR)
    best_defect = MODELS_ROOT / "step3" / "weights" / "best.pt"  
    # 4단계: SAHI 추론 (2단계와 동일 규칙)
    stage4_infer_yolo_with_sahi(
        weights_path=best_defect,
        cropped_test_split=CROP_TEST,   # /test_tiles (images/labels 구조)
        out_dir=STAGE4_DIR,             # 결과 저장 루트
        sahi_cfg=SAHI_CFG,
        keep_empty=True,                # 예측 없을 때도 빈 txt 생성(권장)
        save_vis=True                   # 필요 없으면 False
    )


if __name__ == "__main__":
    main()
