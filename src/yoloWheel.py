# train_yolo.py
import os
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO
# from ultralytics.data.augment import LetterBox  # 필요 시 주석 해제

from config import (
    dataset_yaml_path, YOLO_MODEL, MODEL_DIR, HYPERPARAMS, DATA_DIR,
    GPU_INDEX  # config에서 GPU 인덱스만 활용 (우선순위/CLI 없음)
)

RUN_NAME = "yolo11n"  # 학습 결과 폴더명 통일


def get_device() -> str:
    """
    디바이스 우선순위 제거: CUDA 사용을 강제 시도.
    - config.GPU_INDEX가 지정되면 해당 GPU만 사용
    - CUDA 미가용 시 경고 후 CPU 폴백
    """
    if GPU_INDEX is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        idx = torch.cuda.current_device()
        print(f"✅ Using CUDA device: {torch.cuda.get_device_name(idx)} (index {idx})")
    else:
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"
    return device


class YOLOPipeline:
    def __init__(self, model_path: str, device: str):
        self.device = device
        print(f"Using device: {self.device}")
        if Path(model_path).is_file():
            self.model = YOLO(model_path)
        else:
            print(f"[warn] weights not found: {model_path} -> fallback to 'yolo11n.pt'")
            self.model = YOLO("yolo11n.pt")

    def train(self, data_yaml):
        # img_size = HYPERPARAMS.get("imgsz", 640)
        # self.model.transforms = LetterBox(new_shape=(img_size, img_size), scaleup=False)

        self.model.train(
            data=str(data_yaml),
            project=str(MODEL_DIR),
            name=RUN_NAME,
            device=self.device,   # 'cuda' 또는 'cpu'
            **HYPERPARAMS
        )

    def validate(self, data_yaml, split='val'):
        return self.model.val(data=str(data_yaml), split=split, device=self.device)

    def predict(self, source, save_dir):
        return self.model.predict(
            source=str(source),
            save=True,
            project=str(save_dir),
            name='predict',
            exist_ok=True,
            device=self.device
        )

    def save_metrics(self, results, save_path):
        names = [name for _, name in results.names.items()]
        df = pd.DataFrame({
            'Class': names,
            'Box_AP': results.box.all_ap.mean(axis=1),
            'Box_Precision': results.box.p,
            'Box_Recall': results.box.r
        })
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)


def main():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    device = get_device()
    yolo = YOLOPipeline(YOLO_MODEL, device=device)

    # Train
    print("\n[1/4] Training model...")
    yolo.train(dataset_yaml_path)

    # Validate (val)
    print("\n[2/4] Evaluating validation data...")
    val_results = yolo.validate(dataset_yaml_path, split='val')
    val_box_map = val_results.box.map
    val_fps = 1000.0 / val_results.speed['inference']  # ms/img -> FPS
    print(f"Validation Bounding Box mAP: {val_box_map:.4f}")
    print(f"Validation Inference Speed (FPS): {val_fps:.2f}")

    print("\nClass-wise Performance (Validation - Bounding Box):")
    for i, name in val_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {val_results.box.all_ap.mean(axis=1)[i]:.4f}")
        print(f"  Precision: {val_results.box.p[i]:.4f}")
        print(f"  Recall: {val_results.box.r[i]:.4f}")

    yolo.save_metrics(val_results, Path(MODEL_DIR) / RUN_NAME / "val_results.csv")

    # Validate (test)
    print("\n[3/4] Evaluating test data...")
    test_results = yolo.validate(dataset_yaml_path, split='test')
    test_box_map = test_results.box.map
    test_fps = 1000.0 / test_results.speed['inference']
    print(f"Test Bounding Box mAP: {test_box_map:.4f}")
    print(f"Test Inference Speed (FPS): {test_fps:.2f}")

    print("\nClass-wise Performance (Test - Bounding Box):")
    for i, name in test_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {test_results.box.all_ap.mean(axis=1)[i]:.4f}")
        print(f"  Precision: {test_results.box.p[i]:.4f}")
        print(f"  Recall: {test_results.box.r[i]:.4f}")

    yolo.save_metrics(test_results, Path(MODEL_DIR) / RUN_NAME / "test_results.csv")

    # Predict
    #print("\n[4/4] Predicting on test data...")
    #yolo.predict(Path(DATA_DIR) / "test" / "images", MODEL_DIR)

if __name__ == "__main__":
    main()
