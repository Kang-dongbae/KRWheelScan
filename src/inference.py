# inference.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import os, json, cv2, numpy as np
import shutil


from config import INFER, BUILD_STAGE2, STAGE1, STAGE2, MODEL_DIR, INFER_INPUT_DIR, BUILD_STAGE2_512
from ultralytics import YOLO
from geom_utils import sliding_windows_wh, xyxy_to_xywh

def _mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _read_lines(p: Path): return [] if not p.exists() else p.read_text().strip().splitlines()
def _write_lines(p: Path, lines: List[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(("\n".join(lines) + "\n") if lines else "")

def _yolo_to_xyxy_abs(txt: Path, W:int, H:int) -> List[Tuple[int,int,int,int,int]]:
    out=[]
    for ln in _read_lines(txt):
        sp = ln.split()
        if len(sp)!=5: continue
        cid=int(float(sp[0])); x,y,w,h=map(float, sp[1:])
        cx,cy=x*W,y*H; bw,bh=w*W,h*H
        x1,y1=int(cx-bw/2),int(cy-bh/2); x2,y2=int(cx+bw/2),int(cy+bh/2)
        out.append((cid,x1,y1,x2,y2))
    return out

def _xyxy_abs_to_yolo(x1,y1,x2,y2,W,H,cls)->Optional[str]:
    w=x2-x1; h=y2-y1
    if w<=0 or h<=0: return None
    cx=x1+w/2; cy=y1+h/2
    return f"{cls} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}"

def _area(b): return max(0,b[2]-b[0]) * max(0,b[3]-b[1])
def _inter(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    if x2<=x1 or y2<=y1: return None
    return [x1,y1,x2,y2]

def _pad_roi(xywh, pad_ratio, W,H):
    x,y,w,h=xywh
    px=int(w*pad_ratio); py=int(h*pad_ratio)
    x=max(0,x-px); y=max(0,y-py); w=min(W-x,w+2*px); h=min(H-y,h+2*py)
    return [x,y,w,h]

# ---------------------- Stage-1 inference ----------------------
def infer_stage1():
    cfg = INFER["STAGE1"]
    model = YOLO(cfg["weights"])
    out_dir = Path(cfg["save_dir"]); _mkdir(out_dir)

    model.predict(
        source=str(cfg["source"]), conf=cfg["conf"], iou=cfg["iou"],
        save_txt=cfg["save_txt"], save_crop=cfg["save_crop"],
        project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
        verbose=False
    )
    print(f"[infer_stage1] saved → {out_dir}")

# -------- Stage-2 dataset build: ROI crop + tiling + label remap --------
def build_stage2_tiles():
    
    if BUILD_STAGE2_512.get("six_split", False):
        _build_stage2_six512()
        return
    
    cfg = BUILD_STAGE2
    src_root = Path(cfg["src_root"])
    out_root = Path(cfg["out_root"])
    splits = cfg.get("splits", ("train", "valid", "test"))

    # Stage-1 wheel detector
    wheel = YOLO(cfg["stage1_weights"])
    conf1, iou1 = cfg["conf1"], cfg["iou1"]

    # 크롭 관련 설정
    pad_ratio = float(cfg.get("pad_ratio", 0.10))     # ROI 사각 패딩 비율
    min_cov   = float(cfg.get("min_cov",   0.15))     # GT 상자-ROI 교집합 커버리지 임계
    class_map = cfg["class_map"]                      # 예: {1:0, 2:1}

    for sp in splits:
        img_dir = src_root / sp / "images"
        lbl_dir = src_root / sp / "labels"

        out_img = out_root / f"{sp}_tiles" / "images"
        out_lbl = out_root / f"{sp}_tiles" / "labels"
        _mkdir(out_img); _mkdir(out_lbl)

        for ip in sorted(img_dir.glob("*.*")):
            img = cv2.imread(str(ip))
            if img is None:
                continue
            H, W = img.shape[:2]

            # 1) 휠 검출
            r1 = wheel.predict(
                source=str(ip),
                imgsz=STAGE1["imgsz"],
                conf=conf1,
                iou=iou1,
                verbose=False
            )[0]
            if len(r1.boxes) == 0:
                continue

            xyxy  = r1.boxes.xyxy.cpu().numpy()
            confs = r1.boxes.conf.cpu().numpy()
            if len(xyxy) == 0:
                continue
            best_idx = int(np.argmax(confs))
            wheels = [xyxy[best_idx].astype(int).tolist()]

            if not wheels:
                continue

            # 3) 원본 결함 라벨 로드
            gt = _yolo_to_xyxy_abs(lbl_dir / (ip.stem + ".txt"), W, H)

            # 4) 각 휠 ROI만 '그대로' 크롭 저장 (타일링 없음)
            for wi, wxyxy in enumerate(wheels):
                # 정사각형 패딩 ROI (레일/배경 혼입 완화)
                rx1, ry1, rx2, ry2 = [int(v) for v in wheels[0]]
                rw, rh = rx2 - rx1, ry2 - ry1
                roi_xyxy = [rx1, ry1, rx2, ry2]

                # ROI 내 결함만 추려서 ROI 좌표계로 변환
                yolo_lines = []
                for (cid, x1, y1, x2, y2) in gt:
                    if cid not in class_map:
                        continue
                    inter = _inter([x1, y1, x2, y2], roi_xyxy)
                    if inter is None:
                        continue
                    ia = _area(inter); oa = _area([x1, y1, x2, y2])
                    if oa <= 0 or ia / oa < min_cov:
                        continue
                    ix1, iy1, ix2, iy2 = inter
                    s = _xyxy_abs_to_yolo(ix1 - rx1, iy1 - ry1, ix2 - rx1, iy2 - ry1, rw, rh, class_map[cid])
                    if s:
                        yolo_lines.append(s)

                # 이미지/라벨 저장
                crop = img[ry1:ry2, rx1:rx2]
                outn = f"{ip.stem}_w{wi}.jpg"
                cv2.imwrite(str(out_img / outn), crop)
                _write_lines(out_lbl / (Path(outn).stem + ".txt"), yolo_lines)

        print(f"[build_stage2_crops_only] {sp} → {(out_root / f'{sp}_tiles').as_posix()}")

    # 메타 저장(참고)
    meta = dict(cfg)
    (Path(cfg["out_root"]) / "stage2_build_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print("[build_stage2_crops_only] done.")

#===============================================================================================================

def _build_stage2_six512():
    cfg = BUILD_STAGE2_512
    src_root = Path(cfg["src_root"])
    out_root = Path(cfg["out_root"])
    splits = cfg.get("splits", ("train", "valid",))
    
    pad_mode = cfg.get("pad_mode", "reflect")  # reflect|constant
    six_overlap = float(cfg.get("six_overlap", 0.15))  # 10~20% 권장
    square = int(cfg.get("square_side", 640))
    class_map = cfg["class_map"]
    min_cov = float(cfg.get("min_cov", cfg.get("min_coverage", 0.25)))

    wheel = YOLO(cfg["stage1_weights"])
    conf1, iou1 = cfg["conf1"], cfg["iou1"]

    for sp in splits:
        img_dir = src_root / sp / "images"
        lbl_dir = src_root / sp / "labels"

        out_img = out_root / f"{sp}_tiles2" / "images"
        out_lbl = out_root / f"{sp}_tiles2" / "labels"
        _mkdir(out_img); _mkdir(out_lbl)

        for ip in sorted(img_dir.glob("*.*")):
            img = cv2.imread(str(ip))
            if img is None:
                continue
            H, W = img.shape[:2]

            # 1) Stage-1 휠 ROI (최고 conf 1개)
            r1 = wheel.predict(
                source=str(ip),
                imgsz=STAGE1["imgsz"],
                conf=conf1,
                iou=iou1,
                verbose=False
            )[0]
            if len(r1.boxes) == 0:
                continue
            xyxy = r1.boxes.xyxy.cpu().numpy()
            confs = r1.boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            rx1, ry1, rx2, ry2 = [int(v) for v in xyxy[best_idx]]

            rw, rh = rx2 - rx1, ry2 - ry1
            long_is_x = False       # ← 가로가 더 길어도 '세로 분할'만 수행


            # 2) GT 읽기
            gt = _yolo_to_xyxy_abs(lbl_dir / (ip.stem + ".txt"), W, H)

            # 3) 6분할 타일 좌표 생성 (겹침 포함)
            n = 6
            tiles = []
            if long_is_x:
                unit = rw / n
                stride = max(1, int(round(unit * (1.0 - six_overlap))))
                w_tile = int(round(unit))
                xs = list(range(0, max(1, rw - w_tile + 1), stride)) or [0]
                if xs[-1] != max(0, rw - w_tile):
                    xs.append(max(0, rw - w_tile))
                for tx in xs[:n]:  # n개 근사
                    x1 = rx1 + tx
                    x2 = min(rx2, x1 + w_tile)
                    y1, y2 = ry1, ry2
                    tiles.append((x1, y1, x2, y2))
            else:
                unit = rh / n
                stride = max(1, int(round(unit * (1.0 - six_overlap))))
                h_tile = int(round(unit))
                ys = list(range(0, max(1, rh - h_tile + 1), stride)) or [0]
                if ys[-1] != max(0, rh - h_tile):
                    ys.append(max(0, rh - h_tile))
                for ty in ys[:n]:
                    y1 = ry1 + ty
                    y2 = min(ry2, y1 + h_tile)
                    x1, x2 = rx1, rx2
                    tiles.append((x1, y1, x2, y2))

            # 4) 타일 저장 (비율 유지 스케일 + 정사각 패딩: square x square)
            t_idx = 0
            for (tx1, ty1, tx2, ty2) in tiles:
                tw, th = tx2 - tx1, ty2 - ty1
                if tw <= 0 or th <= 0:
                    continue
                tile_img = img[ty1:ty2, tx1:tx2]
                if tile_img is None or tile_img.size == 0:
                    continue

                # 스케일 & 패딩
                s = square / max(tw, th)
                nw, nh = int(round(tw * s)), int(round(th * s))
                resized = cv2.resize(tile_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
                dw, dh = square - nw, square - nh
                l, r = dw // 2, dw - dw // 2
                t, b = dh // 2, dh - dh // 2
                border = cv2.BORDER_REFLECT_101 if pad_mode == "reflect" else cv2.BORDER_CONSTANT
                padded = cv2.copyMakeBorder(resized, t, b, l, r, border)

                # 라벨 재매핑: 교차 → 타일 좌표 → 스케일/패딩 → YOLO(norm, square x square)
                lines = []
                for (cid, gx1, gy1, gx2, gy2) in gt:
                    if cid not in class_map:
                        continue
                    inter = _inter([gx1, gy1, gx2, gy2], [tx1, ty1, tx2, ty2])
                    if inter is None:
                        continue
                    ia = _area(inter); oa = _area([gx1, gy1, gx2, gy2])
                    if oa <= 0 or ia / oa < min_cov:
                        continue
                    ix1, iy1, ix2, iy2 = inter
                    # 타일 원점 기준
                    ox1, oy1, ox2, oy2 = ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1
                    # 스케일 + 패딩
                    sx1, sy1 = ox1 * s + l, oy1 * s + t
                    sx2, sy2 = ox2 * s + l, oy2 * s + t
                    # YOLO 정규화 (square x square)
                    cx = ((sx1 + sx2) / 2) / square
                    cy = ((sy1 + sy2) / 2) / square
                    ww = (sx2 - sx1) / square
                    hh = (sy2 - sy1) / square
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    ww = min(max(ww, 0.0), 1.0)
                    hh = min(max(hh, 0.0), 1.0)
                    lines.append(f"{class_map[cid]} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

                outn = f"{ip.stem}_w0_six{t_idx:02d}.jpg"
                cv2.imwrite(str(out_img / outn), padded)
                _write_lines(out_lbl / (Path(outn).stem + ".txt"), lines)
                t_idx += 1

        print(f"[_build_stage2_six512] {sp} → {(out_root / f'{sp}_tiles2').as_posix()}")
    print("[_build_stage2_six512] done.")


# ---------------------- Stage-2 inference ----------------------
def infer_stage2():
    cfg = INFER["STAGE2"]
    model = YOLO(cfg["weights"])
    out_dir = Path(cfg["save_dir"]); _mkdir(out_dir)

    if not cfg.get("use_tiling", False):
        model.predict(
            source=str(cfg["source"]), conf=cfg["conf"], iou=cfg["iou"],
            save_txt=cfg["save_txt"],
            project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
            verbose=False
        )
    else:
        imgs = list(Path(cfg["source"]).glob("*.jpg")) + list(Path(cfg["source"]).glob("*.png"))
        for ip in imgs:
            img = cv2.imread(str(ip));  h,w = img.shape[:2]
            tiles = sliding_windows_wh(w,h,cfg["tile"],cfg["overlap"])
            cnt=0
            for (x1,y1,x2,y2) in [(t[0],t[1],t[0]+t[2],t[1]+t[3]) for t in tiles]:
                r = model.predict(img[y1:y2, x1:x2], conf=cfg["conf"], iou=cfg["iou"], verbose=False)[0]
                cnt += len(r.boxes) if hasattr(r,"boxes") else 0
            print(f"[infer_stage2] {ip.name}: {cnt} dets")
    print(f"[infer_stage2] saved → {out_dir}")

# -------------- 두 단계 연결 추론(원본 → 최종 박스) --------------
def infer_two_stage():
    wheel = YOLO(INFER["STAGE1"]["weights"])
    defect = YOLO(INFER["STAGE2"]["weights"])
    images_dir = INFER_INPUT_DIR
    out_dir = MODEL_DIR / "two_stage_infer"; _mkdir(out_dir)

    c1,i1 = STAGE1["conf"], STAGE1["iou"]
    c2,i2 = STAGE2["conf"], STAGE2["iou"]
    tile,ov = STAGE2["tile"], STAGE2["overlap"]
    pad = STAGE2["pad_ratio"]; use_tiling = STAGE2["use_tiling"]

    for ip in sorted(Path(images_dir).glob("*.*")):
        img = cv2.imread(str(ip)); H,W = img.shape[:2]
        r1 = wheel.predict(str(ip), imgsz=STAGE1["imgsz"], conf=c1, iou=i1, verbose=False)[0]
        if len(r1.boxes)==0: continue
        wheels = r1.boxes.xyxy.cpu().numpy().astype(int).tolist()

        canvas = img.copy()
        for wxyxy in wheels:
            rx,ry,rw,rh = _pad_roi(xyxy_to_xywh(wxyxy), pad, W,H)
            crop = img[ry:ry+rh, rx:rx+rw]
            if (not use_tiling) or max(rw,rh)<=tile:
                r2 = defect.predict(crop, imgsz=STAGE2["imgsz"], conf=c2, iou=i2, verbose=False)[0]
                if len(r2.boxes)>0:
                    for (x1,y1,x2,y2) in r2.boxes.xyxy.cpu().numpy().astype(int):
                        cv2.rectangle(canvas, (x1+rx,y1+ry), (x2+rx,y2+ry), (0,255,0), 2)
            else:
                for (tx,ty,tw,th) in sliding_windows_wh(rw,rh,tile,ov):
                    r2 = defect.predict(crop[ty:ty+th, tx:tx+tw], imgsz=STAGE2["imgsz"], conf=c2, iou=i2, verbose=False)[0]
                    if len(r2.boxes)>0:
                        for (x1,y1,x2,y2) in r2.boxes.xyxy.cpu().numpy().astype(int):
                            cv2.rectangle(canvas,(x1+rx+tx,y1+ry+ty),(x2+rx+tx,y2+ry+ty),(0,255,0),2)

        cv2.imwrite(str(out_dir / f"{ip.stem}_pred.jpg"), canvas)
    print(f"[infer_two_stage] saved → {out_dir}")


# ----------------------------- entry -----------------------------
if __name__ == "__main__":
    from config import RUN_MODE
    if RUN_MODE == "infer_stage1":
        infer_stage1()
    elif RUN_MODE == "build_stage2":
        build_stage2_tiles()
    elif RUN_MODE == "infer_stage2":
        infer_stage2()
    elif RUN_MODE == "infer_two_stage":
        infer_two_stage()
    else:
        raise SystemExit(f"inference.py는 RUN_MODE={RUN_MODE} 에서 사용할 수 있는 작업만 수행합니다.")
