import os
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# =========================
# 경로/환경 설정
# =========================
# __file__이 없는 환경(노트북/REPL)에서도 안전하게 동작
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

# 데이터 루트
base_dir = Path('/home/dongbae/Dev/WheelScan/data')
model_dir = ROOT / 'models'

# 분류 학습용 데이터 디렉토리
# - train_defect: 클래스별 하위폴더 구조 필요 (예: train_defect/spalling, train_defect/flat)
# - valid_defect, test_defect 동일
train_defect = base_dir / 'train_defect'
valid_defect = base_dir / 'valid_defect'
test_defect = base_dir / 'test_defect'

# Stage-1 타일(크롭) 평가용 이미지/라벨 디렉토리
test_tiles_dir = base_dir / 'test_tiles'
test_images_dir = test_tiles_dir / 'images'
test_labels_dir = test_tiles_dir / 'labels'

# (선택) 로컬에 최종 저장할 분류 가중치 파일명
final_model_path = model_dir / 'yolo11s-cls'/ 'weights' / 'best.pt'

# 요청하신 저장 경로: model/pred
pred_root = model_dir / 'yolo11s-cls' / 'pred'
overlay_dir = pred_root / 'overlays'
pred_root.mkdir(parents=True, exist_ok=True)
overlay_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 유틸
# =========================
def get_device():
    """가용한 디바이스 반환."""
    return 0 if torch.cuda.is_available() else 'cpu'


def get_ground_truth(label_path: Path) -> str:
    """
    YOLO 감지 형식의 라벨(.txt)에서 이미지 단위 GT를 추출.
    - 파일 없거나 빈 파일: 'normal'
    - 클래스 0만 존재: 'spalling'
    - 클래스 1만 존재: 'flat'
    - 두 클래스 혼재: 'mixed' (평가에서 제외)
    """
    if not label_path.exists():
        return 'normal'

    with open(label_path, 'r') as f:
        lines = f.readlines()

    classes = set()
    for line in lines:
        if line.strip():
            cls = int(line.split()[0])
            classes.add(cls)

    if not classes:
        return 'normal'
    if len(classes) > 1:
        return 'mixed'

    cls = list(classes)[0]
    return 'spalling' if cls == 0 else 'flat'


def safe_font():
    """오버레이용 폰트 로드(기본 폰트 fallback)."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        return ImageFont.load_default()


def draw_overlay(img_path: Path, text: str, out_path: Path):
    """이미지 상단에 텍스트 오버레이 후 저장."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = safe_font()

    # 텍스트 배경 박스
    padding = 6
    text_w, text_h = draw.textlength(text, font=font), font.size + 4
    # PIL의 textlength는 width만 주므로 height는 폰트크기로 근사
    box_w, box_h = int(text_w + padding * 2), int(text_h + padding * 2)
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 160))

    # 텍스트 흰색으로
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    img.save(out_path)


# =========================
# 학습 / 검증
# =========================
def prepare_cls_dataset(cls_root: Path):
    """
    cls_root 아래에 train/val(/test) 심볼릭 링크를 만들어
    Ultralytics 분류 트레이너가 요구하는 디렉토리 구조를 맞춥니다.
    (심볼릭 링크가 안 되는 환경이면 복사)
    """
    cls_root.mkdir(parents=True, exist_ok=True)
    mapping = [('train', train_defect), ('val', valid_defect)]
    if test_defect.exists():
        mapping.append(('test', test_defect))

    for name, src in mapping:
        dst = cls_root / name
        if dst.exists():
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copytree(src, dst)

    return cls_root


def train_model():
    """
    YOLOv11s 분류 모델을 결함 크롭 데이터로 학습.
    - 로컬에 모델 없으면 자동 다운로드
    - best.pt를 지정된 위치로 복사 저장
    """
    project_dir = model_dir / 'yolo11s-cls'
    project_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = prepare_cls_dataset(base_dir / 'cls_dataset')

    model = YOLO("yolo11s-cls.pt")

    # 주의: train()의 val 파라미터는 bool 입니다. (경로 아님)
    results = model.train(
        data=str(dataset_root),   # dataset_root 안에 train/val(/test)
        epochs=100,
        imgsz=224,
        batch=32,
        name='wheel_defect_cls',
        project=str(project_dir),
        exist_ok=True,
        device=get_device(),
        val=True,
    )

    # best.pt 실제 위치: project / name / weights / best.pt
    best_pt = project_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        shutil.copy2(str(best_pt), str(final_model_path))
        print(f"[INFO] 최종 모델 저장: {final_model_path}")
    else:
        print(f"[WARN] best.pt를 찾지 못했습니다: {best_pt}")

    print(f"[INFO] 학습 완료! 결과 위치: {project_dir}")
    return model


def validate_model(model: YOLO):
    """검증 세트(valid_defect) 평가."""
    if not valid_defect.exists():
        print(f"[WARN] 검증 폴더를 찾을 수 없습니다: {valid_defect}")
        return None

    results = model.val(data=str(valid_defect))
    print("[INFO] Validation results:")
    print(results)
    return results


def test_model(model: YOLO):
    """테스트 세트(test_defect) 평가(옵션)."""
    if not test_defect.exists():
        print(f"[WARN] 테스트 폴더가 없습니다: {test_defect}")
        return None

    results = model.val(data=str(test_defect))
    print("[INFO] Test results:")
    print(results)
    return results


# =========================
# Stage-1 타일 평가 + 저장
# =========================
def save_confusion_matrix_png(cm: np.ndarray, class_names, out_path: Path, title="Confusion Matrix"):
    """혼동행렬 PNG 저장."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='GT', xlabel='Pred', title=title)

    # 값 표기
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
def _get_label_by_index(model, idx: int):
    """model.names가 dict/list 어느 쪽이든 인덱스로 라벨명 얻기."""
    names = getattr(model, 'names', None)
    if names is None:
        return None
    if isinstance(names, dict):
        return names.get(idx)
    if isinstance(names, list):
        if 0 <= idx < len(names):
            return names[idx]
    return None


def _get_index_by_label(model, target: str):
    """model.names에서 라벨명 -> 인덱스 역탐색 (없으면 None)."""
    names = getattr(model, 'names', None)
    if names is None:
        return None
    if isinstance(names, dict):
        for k, v in names.items():
            if v == target:
                return int(k)
    if isinstance(names, list):
        for i, v in enumerate(names):
            if v == target:
                return i
    return None

def test_on_stage1_crops_and_save(model: YOLO, conf_threshold: float = 0.5, out_dir: Path = pred_root):
    """
    Stage-1 크롭 이미지에 대해 분류 추론 수행 + 결과 저장
    - conf < threshold면 'normal'로 처리(오픈셋 거부)
    - 저장물:
      * out_dir/preds.csv : 이미지별 GT/Pred/확신도/클래스 확률
      * out_dir/summary.txt : 정확도 등 요약
      * out_dir/confusion_matrix.png : 혼동행렬 이미지
      * out_dir/overlays/*.jpg : 예측/정답 오버레이 이미지
    - 모델이 ImageNet 클래스 등을 내뱉을 경우 허용 외 라벨은 'normal'로 강제 매핑
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'overlays').mkdir(parents=True, exist_ok=True)

    if not test_images_dir.exists():
        raise FileNotFoundError(f"[ERROR] Test tiles images directory not found: {test_images_dir}")

    # 혼동행렬용 클래스 & 허용 결함 라벨
    cm_classes = ['normal', 'spalling', 'flat']
    class_to_idx = {c: i for i, c in enumerate(cm_classes)}
    allowed_defect_labels = set(cm_classes) - {'normal'}

    # 모델 클래스 디버그
    try:
        print(f"[DEBUG] model.names length = {len(model.names) if hasattr(model, 'names') else 'NA'}")
    except Exception:
        pass

    filenames, ground_truths, predictions, confidences = [], [], [], []
    probs_list = []
    correct = 0
    total = 0

    image_files = sorted(
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not image_files:
        print(f"[WARN] 평가할 이미지가 없습니다: {test_images_dir}")
        return

    # 모델에서 spalling/flat 인덱스(있다면) 찾기
    idx_spalling = _get_index_by_label(model, 'spalling')
    idx_flat = _get_index_by_label(model, 'flat')

    for img_file in image_files:
        img_path = test_images_dir / img_file
        base_name = Path(img_file).stem
        label_path = test_labels_dir / f"{base_name}.txt"

        # GT
        gt = get_ground_truth(label_path)

        # 추론
        results = model(str(img_path), imgsz=224)
        probs = results[0].probs  # Classification probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf.item())

        pred_name = _get_label_by_index(model, top1_idx)

        # 최종 예측 라벨 결정 (거부+허용집합 매핑)
        if top1_conf < conf_threshold:
            pred = 'normal'
        else:
            if pred_name in allowed_defect_labels:
                pred = pred_name
            else:
                pred = 'normal'  # 허용 외 라벨은 normal로

        # 정답 집계
        if gt == pred:
            correct += 1
        total += 1

        # 오버레이
        overlay_text = f"Pred: {pred} ({top1_conf:.2f}) | GT: {gt}"
        draw_overlay(img_path, overlay_text, overlay_dir / img_file)

        # per-class 확률 (우리 두 클래스만 안전하게 기록)
        ps = float(probs.data[idx_spalling].item()) if idx_spalling is not None else np.nan
        pf = float(probs.data[idx_flat].item()) if idx_flat is not None else np.nan

        # 누적
        filenames.append(img_file)
        ground_truths.append(gt)
        predictions.append(pred)
        confidences.append(top1_conf)
        probs_list.append((ps, pf))

        # 콘솔 로그
        print(f"Image: {img_file} | GT: {gt} | Pred: {pred} (conf: {top1_conf:.2f})")

    # 정확도
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n[RESULT] Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    # 혼동행렬 (GT 'mixed'는 제외)
    cm = np.zeros((len(cm_classes), len(cm_classes)), dtype=int)
    for gt, pred in zip(ground_truths, predictions):
        if gt == 'mixed':
            continue
        gt_idx = class_to_idx[gt] if gt in class_to_idx else class_to_idx['normal']
        pred_idx = class_to_idx[pred] if pred in class_to_idx else class_to_idx['normal']
        cm[gt_idx, pred_idx] += 1

    # 저장물 1) CSV
    csv_path = out_dir / "preds.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt", "pred", "conf_top1", "p_spalling", "p_flat"])
        for fn, gt, pred, conf, (ps, pf) in zip(filenames, ground_truths, predictions, confidences, probs_list):
            writer.writerow([fn, gt, pred, f"{conf:.6f}", f"{ps:.6f}", f"{pf:.6f}"])

    # 저장물 2) summary.txt
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")
        f.write("Confusion Matrix (Rows=GT, Cols=Pred)\n")
        f.write(f"Classes: {cm_classes}\n")
        f.write(np.array2string(cm, separator=' '))

    # 저장물 3) 혼동행렬 PNG
    cm_png = out_dir / "confusion_matrix.png"
    save_confusion_matrix_png(cm, cm_classes, cm_png, title="Confusion Matrix (GT vs Pred)")

    print(f"[INFO] 저장 완료")
    print(f" - CSV: {csv_path}")
    print(f" - Summary: {summary_path}")
    print(f" - Confusion Matrix PNG: {cm_png}")
    print(f" - Overlays: {overlay_dir}/*.jpg")


# =========================
# 엔트리포인트
# =========================
if __name__ == "__main__":
    # 1) 학습 (필요 시)
    model = train_model()

    # 이미 학습된 가중치가 있으면 로드 (권장)
    # - final_model_path 또는 best.pt 경로를 지정하세요.
    if final_model_path.exists():
        model = YOLO(str(final_model_path))
        print(f"[INFO] 로드: {final_model_path}")
    else:
        # 없다면 사전학습 가중치로 시작(정확도 낮을 수 있음)
        model = YOLO("yolo11s-cls.pt")
        print("[WARN] 로컬 최종 가중치를 찾지 못했습니다. yolo11s-cls.pt로 진행합니다.")

    # 2) 검증 세트 평가(옵션)
    # validate_model(model)

    # 3) Stage-1 크롭 이미지로 테스트 + 저장
    # 필요 시 임계치 조정 (0.5~0.8 권장 탐색)
    test_on_stage1_crops_and_save(model, conf_threshold=0.5)
