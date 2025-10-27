# advancedYolo.py
from __future__ import annotations
import os, json, shutil, glob
from pathlib import Path
from typing import List, Tuple, Dict

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from config import (
    ROOT, DATA_DIR, MODEL_DIR, RUN_MODE,
    STAGE1, STAGE2,
)

# train()에 안전하게 넘길 수 있는 키만 필터
# ALLOWED_TRAIN_ARGS 보강

def _mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _save_cfg(out_dir: Path, obj: dict, name: str):
    _mkdir(out_dir); (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")

# Stage-1용 wheel-only 라벨 미러 데이터셋 구성
def _build_stage1_mirror_yaml() -> Path:
    mirror_root = MODEL_DIR / "_stage1_ds"
    if mirror_root.exists():
        shutil.rmtree(mirror_root)
    for sp in ("train","valid","test"):
        (mirror_root / sp / "images").mkdir(parents=True, exist_ok=True)

    def _link_or_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except Exception:
            if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
            else: shutil.copy2(src, dst)

    # images: 원본
    for sp in ("train","valid","test"):
        _link_or_copy(DATA_DIR / sp / "images", mirror_root / sp / "images")

    # labels: wheel-only
    for sp in ("train","valid","test"):
        src = DATA_DIR / sp / "labels_stage1"
        if not src.exists():
            raise FileNotFoundError(f"[Stage-1] wheel-only 라벨이 필요합니다: {src}")
        _link_or_copy(src, mirror_root / sp / "labels")

    yml = "\n".join([
        "train: train/images",
        "val: valid/images",
        "",
        "nc: 1",
        "names: ['wheel']",
    ])
    (mirror_root / "stage1.yaml").write_text(yml, encoding="utf-8")
    return mirror_root / "stage1.yaml"

def train_stage1():
    if YOLO is None: raise RuntimeError("Install 'ultralytics' first.")
    out_dir = Path(STAGE1["save_dir"]).resolve(); _mkdir(out_dir)
    stage1_yaml = _build_stage1_mirror_yaml()

    model = YOLO(STAGE1.get("weights", str(ROOT / "yolo11s.pt")))
    #args = {k:v for k,v in STAGE1.items() if k in ALLOWED_TRAIN_ARGS}
    #args.setdefault("plots", STAGE1.get("plots", True))

    model.train(
        data=str(stage1_yaml),
        project=str(out_dir), name="", exist_ok=True,
        **STAGE1
    )

    try:
        from ultralytics.utils.plotting import plot_results
        res_csv = out_dir / "results.csv"
        if res_csv.exists(): plot_results(file=res_csv)
    except Exception:
        pass
    print(f"[stage1] done → {out_dir}")

def train_stage2():

    base_out = Path(STAGE2["save_dir"]).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    train_args = STAGE2.copy()
    model_path = train_args.pop("model")
    #weights_path = train_args.pop("weights") # None 일 경우 초기화에 사용됨

    model = YOLO(model_path) 
    
    model.train( 
        data=train_args.pop("data_yaml"), # data_yaml도 분리하여 명시적으로 전달
        project=str(base_out),
        name="stage2", 
        #optimizer='SGD',  
        **train_args      # 나머지 학습/HP 인자 전달
    )

    # --- ⑤ 결과 플롯 ---
    try:
        from ultralytics.utils.plotting import plot_results
        res_csv = base_out / "results.csv"
        if res_csv.exists():
            plot_results(file=res_csv)
    except Exception:
        pass

    print(f"[stage2] done → {base_out}")

def main():
    mode = RUN_MODE
    if mode == "train_stage1":
        train_stage1()
    elif mode == "train_stage2":
        train_stage2()
    else:
        raise SystemExit(f"advancedYolo.py는 학습 전용입니다. RUN_MODE={mode} 는 inference.py에서 실행하세요.")

if __name__ == "__main__":
    main()
