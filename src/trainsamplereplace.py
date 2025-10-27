import os
import shutil
import random
from tqdm import tqdm

######### train data 만 데이터 셋 조정 정상 결함 비율을 2:1로 맞춤 #########

# 원본 경로 설정
img_dir = "/home/dongbae/Dev/WheelScan/data/train_tiles2/images"       # 이미지 폴더
label_dir = "/home/dongbae/Dev/WheelScan/data/train_tiles2/labels"     # 라벨 폴더

# 저장 경로 설정
out_img_dir = "/home/dongbae/Dev/WheelScan/data/train_tiles_rand/images"
out_label_dir = "/home/dongbae/Dev/WheelScan/data/train_tiles_rand/labels"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

# 라벨 파일 목록 불러오기
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# 결함 여부 분류
def has_defect(label_path):
    """YOLO 형식 라벨에서 객체가 하나라도 있으면 결함으로 간주"""
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return len(lines) > 0  # 빈 파일이면 정상(결함 없음)

defect_labels = []
normal_labels = []

for lf in tqdm(label_files, desc="라벨 분류 중"):
    lp = os.path.join(label_dir, lf)
    if has_defect(lp):
        defect_labels.append(lf)
    else:
        normal_labels.append(lf)

print(f"결함 타일: {len(defect_labels)}, 정상 타일: {len(normal_labels)}")

# 샘플링 비율 설정 (1:1 or 1:2)
RATIO = 2  # 1:1이면 1, 1:2면 2
sampled_normals = random.sample(normal_labels, min(len(normal_labels), len(defect_labels) * RATIO))

print(f"선택된 정상 타일 수: {len(sampled_normals)}")

# 최종 학습용 리스트
selected_labels = defect_labels + sampled_normals

# 이미지/라벨 복사
for lf in tqdm(selected_labels, desc="파일 복사 중"):
    img_name = lf.replace(".txt", ".jpg")  # 확장자 변경 (jpg/png이면 수정)
    src_img = os.path.join(img_dir, img_name)
    src_label = os.path.join(label_dir, lf)

    dst_img = os.path.join(out_img_dir, img_name)
    dst_label = os.path.join(out_label_dir, lf)

    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)
    else:
        print(f"⚠️ 이미지 없음: {src_img}")

print("✅ 샘플링 완료!")
print(f"총 {len(selected_labels)}개의 이미지와 라벨이 {out_img_dir} 에 저장되었습니다.")
