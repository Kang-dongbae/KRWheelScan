#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# 프로젝트 설정 불러오기 (DEFECTUSE, 경로 등)
import config

# ----- 고정 설정 -----
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
PAD_VAL = 114
# 6→3 클래스 매핑: old(0 crack,1 discolor,2 flat,3 shelling,4 spalling,5 wheel) -> new(0 wheel,1 spalling,2 flat)
CLASS_MAP = {5: 0, 4: 1, 3: 1, 2: 2, 1: None, 0: None}
# L/R 서브셋에만 적용되는 분할 비율
R_TRAIN, R_VAL, R_TEST = 0.7, 0.15, 0.15
# ---------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_empty_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*"):
        if p.is_file():
            # Python 3.8+: missing_ok
            p.unlink(missing_ok=True)
        else:
            shutil.rmtree(p, ignore_errors=True)


def _iter_images(dir_path: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted(dir_path.glob(f"*{ext}")))
    return files


def _tile_positions(total, tile, stride):
    """이미지 하단/우측 경계까지 반드시 덮도록 마지막 포지션 강제 포함."""
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def read_yolo_labels(txt_path: Path):
    labs = []
    if not txt_path.is_file():
        return labs
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            ps = ln.split()
            if len(ps) != 5:
                continue
            try:
                c, cx, cy, w, h = int(ps[0]), float(ps[1]), float(ps[2]), float(ps[3]), float(ps[4])
            except Exception:
                continue
            labs.append([c, cx, cy, w, h])
    return labs


def write_yolo_labels(txt_path: Path, labs):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for c, cx, cy, w, h in labs:
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def clean_labels_and_copy(src_img_dir: Path, src_lab_dir: Path,
                          clean_img_dir: Path, clean_lab_dir: Path,
                          exclude_prefixes=None):
    """
    6→3 리맵, 비어있는 샘플 제외, clean 세트 생성
    exclude_prefixes: ['D'] 등 접두어 리스트. 해당 접두어로 시작하는 stem은 전처리 대상에서 제외.
    return: (stems_list, stems_classes_dict)
    """
    clean_img_dir.mkdir(parents=True, exist_ok=True)
    clean_lab_dir.mkdir(parents=True, exist_ok=True)

    kept = dropped = skipped_by_prefix = 0
    before_boxes = after_boxes = 0
    cls_cnt = Counter()
    stems_classes = {}

    for lab in sorted(src_lab_dir.glob("*.txt")):
        stem = lab.stem

        # 접두어 제외 규칙 적용 (e.g., DEFECTUSE=False일 때 D* 제외)
        if exclude_prefixes:
            up = stem.upper()
            if any(up.startswith(pref.upper()) for pref in exclude_prefixes):
                skipped_by_prefix += 1
                continue

        # 원본 이미지 탐색(다중 확장자 대응)
        img = None
        for ext in IMG_EXTS:
            cand = src_img_dir / f"{stem}{ext}"
            if cand.is_file():
                img = cand
                break
        if img is None:
            continue

        with open(lab, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        before_boxes += len(lines)

        new_lines, new_cls = [], []
        for ln in lines:
            ps = ln.split()
            try:
                cid = int(ps[0])
            except Exception:
                continue
            nc = CLASS_MAP.get(cid, None)
            if nc is None:
                continue
            ps[0] = str(nc)
            new_lines.append(" ".join(ps))
            new_cls.append(nc)

        if not new_lines:
            dropped += 1
            continue

        shutil.copy2(img, clean_img_dir / img.name)
        with open(clean_lab_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        kept += 1
        after_boxes += len(new_lines)
        stems_classes[stem] = new_cls
        cls_cnt.update(new_cls)

    print("=== Clean Summary ===")
    print(f"kept: {kept}, dropped(empty after remap): {dropped}, skipped_by_prefix: {skipped_by_prefix}")
    print(f"boxes: {before_boxes} -> {after_boxes}")
    print(f"class dist (0:wheel,1:spalling,2:flat): {dict(cls_cnt)}")
    return list(stems_classes.keys()), stems_classes


def split_LR_712(stems_LR, stems_classes):
    """
    L/R만 대상으로 major class 기준 간단 계층 분할.
    각 클래스 그룹별로 7:1:2를 적용, 합쳐서 최종 LR 분할 생성.
    """
    from collections import Counter as Cnt
    per_class = defaultdict(list)
    for s in stems_LR:
        major = Cnt(stems_classes[s]).most_common(1)[0][0]
        per_class[major].append(s)

    tr, va, te = [], [], []
    for _, arr in per_class.items():
        random.shuffle(arr)
        n = len(arr)
        n_train = round(n * R_TRAIN)
        n_val   = round(n * R_VAL)
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = min(n_val, n - n_train)
        tr += arr[:n_train]
        va += arr[n_train:n_train+n_val]
        te += arr[n_train+n_val:]

    random.seed(42)  # seed 고정
    random.shuffle(tr); random.shuffle(va); random.shuffle(te)
    print("=== LR Split Summary (7:1:2 on LR only) ===")
    print(f"LR-train: {len(tr)}, LR-val: {len(va)}, LR-test: {len(te)} (LR total {len(stems_LR)})")
    return tr, va, te


def copy_split(name, stems, src_img_dir, src_lab_dir, dst_img_dir, dst_lab_dir):
    ensure_empty_dir(dst_img_dir)
    ensure_empty_dir(dst_lab_dir)
    kept = 0
    for s in stems:
        img_src = None
        for ext in IMG_EXTS:
            cand = src_img_dir / f"{s}{ext}"
            if cand.is_file():
                img_src = cand
                break
        lab_src = src_lab_dir / f"{s}.txt"
        if not (img_src and lab_src.is_file()):
            continue
        shutil.copy2(img_src, dst_img_dir / img_src.name)   # 확장자 보존
        shutil.copy2(lab_src,  dst_lab_dir  / lab_src.name)
        kept += 1
    print(f"[{name}] copied: {kept}")


# ---------- 규격화(학습용 train에만 적용) ----------

def pad_only_and_fix_labels(img, labels, final_size=1024, pad_val=PAD_VAL):
    """
    업스케일 금지 패딩-only + 라벨 좌표 보정(+클리핑).
    입력 라벨은 원본 W,H 정규화 기준 → 패딩 후 final_size 기준으로 재정규화.
    """
    H, W = img.shape[:2]
    canvas = np.full((final_size, final_size, 3), pad_val, dtype=img.dtype)

    # scaleup 금지
    scale = min(final_size / max(W, 1), final_size / max(H, 1))
    if scale > 1.0:
        scale = 1.0

    new_w, new_h = int(W * scale), int(H * scale)
    img_resized = img if (new_h, new_w) == (H, W) else cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top  = (final_size - new_h) // 2
    left = (final_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized

    fixed = []
    for c, cx, cy, w, h in labels:
        cxp = cx * W * scale + left
        cyp = cy * H * scale + top
        wp  = w  * W * scale
        hp  = h  * H * scale
        nx, ny = cxp / final_size, cyp / final_size
        nw, nh = wp  / final_size, hp  / final_size

        # 수치 안전장치(클리핑)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        if nw <= 0 or nh <= 0:
            continue
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            continue
        fixed.append([c, nx, ny, nw, nh])

    return canvas, fixed


def tile_labels_for_window(img_wh, tile_xywh, labels, min_box_px=2):
    """
    원본 정규화 라벨 → 타일 윈도우와 교차 영역을 타일 기준으로 재정규화.
    """
    W, H = img_wh
    tx, ty, tw, th = tile_xywh
    out = []
    for (cls, cx, cy, bw, bh) in labels:
        cxp, cyp = cx*W, cy*H
        wp, hp = bw*W, bh*H
        x1, y1 = cxp - wp/2, cyp - hp/2
        x2, y2 = cxp + wp/2, cyp + hp/2

        ix1, iy1 = max(x1, tx), max(y1, ty)
        ix2, iy2 = min(x2, tx+tw), min(y2, ty+th)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        lx1, ly1 = ix1 - tx, iy1 - ty
        lx2, ly2 = ix2 - tx, iy2 - ty

        if (lx2 - lx1) < min_box_px or (ly2 - ly1) < min_box_px:
            continue

        lcx = (lx1 + lx2) / 2 / tw
        lcy = (ly1 + ly2) / 2 / th
        lw  = (lx2 - lx1) / tw
        lh  = (ly2 - ly1) / th
        if lw <= 0 or lh <= 0:
            continue
        if not (0.0 <= lcx <= 1.0 and 0.0 <= lcy <= 1.0):
            continue
        out.append([cls, lcx, lcy, lw, lh])
    return out


def save_image(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def normalize_train_to_1024(
    train_root: Path,
    final_size=1024,
    tile=1024,
    overlap=0.20,
    remove_empty_tiles=True,
    min_box_px=2
):
    """
    train_root: data/train
      - max(H,W) ≤ final_size → 패딩-only(+라벨 보정) → data/train_1024
      - max(H,W) >  final_size → 타일링(+overlap), 경계 타일 보장, 빈 타일 제외 → data/train_1024
    valid/test는 변형하지 않음.
    """
    in_img = train_root / "images"
    in_lab = train_root / "labels"
    out_root = Path(str(train_root) + f"_{final_size}")
    out_img = out_root / "images"
    out_lab = out_root / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    stride = int(tile * (1.0 - overlap))
    total, saved = 0, 0

    for p in _iter_images(in_img):
        total += 1
        stem = p.stem
        labp = in_lab / f"{stem}.txt"

        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        labels = read_yolo_labels(labp)

        if max(H, W) <= final_size:
            canvas, fixed = pad_only_and_fix_labels(img, labels, final_size=final_size, pad_val=PAD_VAL)
            out_name = f"{stem}{p.suffix}"
            save_image(out_img / out_name, canvas)
            write_yolo_labels(out_lab / f"{stem}.txt", fixed)
            saved += 1
            continue

        # 타일링(경계 타일 포함)
        ys = _tile_positions(H, tile, stride)
        xs = _tile_positions(W, tile, stride)
        tid = 0
        for y in ys:
            for x in xs:
                crop = img[y:y+tile, x:x+tile]
                ch, cw = crop.shape[:2]
                if ch < tile or cw < tile:
                    canvas = np.full((tile, tile, 3), PAD_VAL, dtype=img.dtype)
                    canvas[:ch, :cw] = crop
                    crop = canvas
                tlabels = tile_labels_for_window((W, H), (x, y, tile, tile), labels, min_box_px=min_box_px)
                if remove_empty_tiles and len(tlabels) == 0:
                    continue
                out_stem = f"{stem}_t{tid:03d}"
                out_name = f"{out_stem}{p.suffix}"
                save_image(out_img / out_name, crop)
                write_yolo_labels(out_lab / f"{out_stem}.txt", tlabels)
                tid += 1

        saved += 1

    print(f"[normalize] train: processed {saved}/{total} images → {out_root}")
    return out_root


# -------------- 메인 파이프라인 --------------

def main():
    ap = argparse.ArgumentParser()
    # src 경로는 인자를 주지 않으면 config.ALL_IMAGES/{images,labels} 사용
    ap.add_argument("--src-img",   type=str, default=None)
    ap.add_argument("--src-lab",   type=str, default=None)
    # clean/출력 경로는 기본값 제공(원하면 인자로 덮어쓰기 가능)
    ap.add_argument("--clean-img", type=str, default=str(config.DATA_DIR / "allimg_clean" / "images"))
    ap.add_argument("--clean-lab", type=str, default=str(config.DATA_DIR / "allimg_clean" / "labels"))
    ap.add_argument("--out-train", type=str, default=str(config.DATA_DIR / "train"))
    ap.add_argument("--out-valid", type=str, default=str(config.DATA_DIR / "valid"))
    ap.add_argument("--out-test",  type=str, default=str(config.DATA_DIR / "test"))
    ap.add_argument("--final-size", type=int, default=1024)
    ap.add_argument("--tile",       type=int, default=1024)
    ap.add_argument("--overlap",    type=float, default=0.20)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--purge-outputs", type=int, default=1)
    # train 타일에서 빈 타일 제외(True 권장)
    ap.add_argument("--remove-empty-tiles", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)

    # 소스 경로: 인자가 없으면 config 기준(default)
    SRC_IMG_DIR   = Path(args.src_img) if args.src_img else (config.ALL_IMAGES / "images")
    SRC_LAB_DIR   = Path(args.src_lab) if args.src_lab else (config.ALL_IMAGES / "labels")
    CLEAN_IMG_DIR = Path(args.clean_img)
    CLEAN_LAB_DIR = Path(args.clean_lab)

    TRAIN_ROOT = Path(args.out_train)
    VAL_ROOT   = Path(args.out_valid)
    TEST_ROOT  = Path(args.out_test)

    TRAIN_IMG_DIR = TRAIN_ROOT / "images"
    TRAIN_LAB_DIR = TRAIN_ROOT / "labels"
    VAL_IMG_DIR   = VAL_ROOT / "images"
    VAL_LAB_DIR   = VAL_ROOT / "labels"
    TEST_IMG_DIR  = TEST_ROOT / "images"
    TEST_LAB_DIR  = TEST_ROOT / "labels"

    # defectuse는 옵션 미사용 → config에서만 읽음
    defectuse = bool(getattr(config, "DEFECTUSE", True))
    exclude_prefixes = None if defectuse else ('D',)
    print("=== Config ===")
    print(f"DEFECTUSE (from config.py): {defectuse} -> exclude_prefixes={exclude_prefixes}")
    print(f"SRC_IMG_DIR: {SRC_IMG_DIR}")
    print(f"SRC_LAB_DIR: {SRC_LAB_DIR}")

    # 1) 리맵 + clean 카피 (필요 시 접두어 제외 적용)
    stems, stems_classes = clean_labels_and_copy(
        SRC_IMG_DIR, SRC_LAB_DIR,
        CLEAN_IMG_DIR, CLEAN_LAB_DIR,
        exclude_prefixes=exclude_prefixes
    )

    # 2) 접두어 그룹
    stems_D  = [s for s in stems if s[:1].upper() == 'D']
    stems_LR = [s for s in stems if s[:1].upper() in ('L', 'R')]
    print("=== Prefix Groups ===")
    print(f"D-only (train): {len(stems_D)}")
    print(f"LR (split 7:1:2): {len(stems_LR)}")

    # 3) L/R 7:1:2 분할
    lr_tr, lr_va, lr_te = split_LR_712(stems_LR, stems_classes)

    # 4) 최종 세트 구성
    #    - DEFECTUSE=True: D*는 train 전용 + LR은 7:1:2
    #    - DEFECTUSE=False: 입력 단계에서 D*를 제외했으므로 train은 LR-tr만 포함
    train = (stems_D if defectuse else []) + lr_tr
    val   = lr_va
    test  = lr_te

    # 5) 분할 복사
    if args.purge_outputs:
        ensure_empty_dir(TRAIN_IMG_DIR); ensure_empty_dir(TRAIN_LAB_DIR)
        ensure_empty_dir(VAL_IMG_DIR);   ensure_empty_dir(VAL_LAB_DIR)
        ensure_empty_dir(TEST_IMG_DIR);  ensure_empty_dir(TEST_LAB_DIR)

    copy_split("train", train, CLEAN_IMG_DIR, CLEAN_LAB_DIR, TRAIN_IMG_DIR, TRAIN_LAB_DIR)
    copy_split("val",   val,   CLEAN_IMG_DIR, CLEAN_LAB_DIR, VAL_IMG_DIR,   VAL_LAB_DIR)
    copy_split("test",  test,  CLEAN_IMG_DIR, CLEAN_LAB_DIR, TEST_IMG_DIR,  TEST_LAB_DIR)

    print("\n✅ Split Done.")
    print(" - Cleaned base : {}".format((config.DATA_DIR / 'allimg_clean')))
    print(" - Splits       : {}, {}, {}".format(TRAIN_ROOT, VAL_ROOT, TEST_ROOT))
    if defectuse:
        print(" - Rule         : (ALL) D → train only; L/R → 7:1:2")
    else:
        print(" - Rule         : (D 제외) L/R만 7:1:2")

    # 6) train만 규격화(패딩-only/타일링) → data/train_1024 생성
    train_1024 = normalize_train_to_1024(
        TRAIN_ROOT,
        final_size=args.final_size,
        tile=args.tile,
        overlap=args.overlap,
        remove_empty_tiles=bool(args.remove_empty_tiles),
        min_box_px=2
    )

    print("\n✅ Train 규격화 완료:")
    print(f" - train: {train_1024}")
    print("\n학습 data.yaml을 아래처럼 설정하세요:")
    print("train: {}/images".format(train_1024))
    print("val:   {}/images".format(VAL_ROOT))
    print("test:  {}/images".format(TEST_ROOT))
    print("nc: 3")
    print("names: ['wheel','spalling','flat']")


if __name__ == "__main__":
    main()
