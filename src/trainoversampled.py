from pathlib import Path

# ==== 경로 세팅 ====
DATA_DIR = Path("/home/dongbae/Dev/WheelScan/data/train_tiles")
IMG_DIR  = DATA_DIR / "images"
LAB_DIR  = DATA_DIR / "labels"

OUT_TXT = DATA_DIR / "train_oversampled.txt"

SPALLING_CLS = 0   # YOLO 라벨에서 spalling class id (필요 시 수정)

def has_spalling(lbl_path):
    """해당 라벨 파일에 spalling 클래스가 있으면 True"""
    with open(lbl_path) as f:
        for line in f:
            if not line.strip():
                continue
            cid = int(float(line.split()[0]))
            if cid == SPALLING_CLS:
                return True
    return False

# ==== 전체 이미지 순회 ====
spall, other, background = [], [], []

for lbl_path in LAB_DIR.glob("*.txt"):
    stem = lbl_path.stem
    img_path = (IMG_DIR / f"{stem}.jpg")
    if not img_path.exists():
        img_path = (IMG_DIR / f"{stem}.png")
    if not img_path.exists():
        continue

    if lbl_path.stat().st_size == 0:
        background.append(str(img_path))
        continue

    if has_spalling(lbl_path):
        spall.append(str(img_path))
    else:
        other.append(str(img_path))

# oversampling 비율
out_list = spall * 3 + other + background[:int(len(background) * 0.3)]
OUT_TXT.write_text("\n".join(out_list))

print(f"✅ Oversampled train list saved to {OUT_TXT}")
print(f"spalling:{len(spall)}, other:{len(other)}, background:{len(background)}, total:{len(out_list)}")
