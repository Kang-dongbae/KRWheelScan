# config.py
from pathlib import Path

# 디바이스 선택: "auto" | "cuda" | "mps" | "cpu"
DEVICE = "auto"        # 기본은 자동판단
GPU_INDEX = None       # 특정 GPU 인덱스 강제하려면 0,1,... 로 지정

ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트 기준(필요시 조정)
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TEST_FOLDER = DATA_DIR / "test" / "images"
dataset_yaml_path = DATA_DIR / "data.yaml"

ALL_IMAGES = DATA_DIR / "AllImg"

YOLO_MODEL = "yolo11n.pt"  
#latest_run = sorted(MODEL_DIR.glob("yolo11n*"))[-1]
#YOLO_MODEL = latest_run / "weights" / "best.pt" 

STAGE1_WEIGHTS = MODEL_DIR / "stage1" / "weights" / "best.pt"
#STAGE1_WEIGHTS = "yolo11s.pt"  

STAGE2_WEIGHTS = "yolo11s.pt"
#STAGE2_WEIGHTS = MODEL_DIR / "stage2" / "weights" / "best.pt"

CLASSES = ['wheel', 'spalling', 'flat']

# 이미지 분류할 떄 결함이미지만 학습으로 사용할지 말지 설정 (휠 검출에는 필요 없음)
DEFECTUSE = False

HYPERPARAMS = dict(
    epochs=50,            
    imgsz=1280,
    batch=12,            
    workers=8,
    cache=True,
    patience=40,
    deterministic=False,
    amp=True,
    val=True,
    optimizer='AdamW',
    rect=False,

    cls=1.0,
    lr0=0.007,

    mosaic=0.15,
    close_mosaic=10,
    mixup=0.05,
    copy_paste=0.20,
    erasing=0.10,

    scale=0.0,

    hsv_h=0.20, hsv_s=0.35, hsv_v=0.25,
    fliplr=0.5, flipud=0.0,
    degrees=10, translate=0.10, shear=1.5, perspective=0.0008,
)



# 학습결과 보고 mosaic 0.6 0.7 erasing 0.30 완화, 

# ====================== STAGE 설정 ======================
STAGE1 = dict(
    data_yaml=str(DATA_DIR / "stage1.yaml"),
    weights=str(STAGE1_WEIGHTS),     # 11s 사용

    imgsz=1280,           
    epochs=100,            
    batch=20,            
    seed=42,
    patience=25,         

    save_dir=str(MODEL_DIR / "stage1"),

    conf=0.50,            
    iou=0.50,            

    amp=True,
    rect=True,
    workers=10,
    cache=True,
    deterministic=False,  
    plots=False,
    verbose=False,

    mosaic=0.2,
    mixup=0.0,
    erasing=0.05,
    close_mosaic=10,

    optimizer='AdamW',
    lr0=0.005,
    freeze=10            
)


# config.py
# -------- Stage-2 (개선 반영) ----------
#STAGE2_WEIGHTS = ROOT / "yolo11s.pt"  # 처음은 ImageNet/COCO 프리트레인
STAGE2 = dict(
    data_yaml=str(DATA_DIR / "stage2.yaml"),  # 위에서 만든 oversampled txt 사용
    model=str(ROOT / "yolo11s-p2.yaml"),  # 여유되면 s→m로 캐파↑ (12GB면 imgsz 1280~1536 배치 조절)
    #weights=None,          # 스크래치
    save_dir=str(MODEL_DIR),

    optimizer='SGD',
    imgsz=640,
    rect=False,            
    epochs=200,
    batch=24,
    workers=8,              
    patience=30,

    lr0=0.01,      
    lrf=0.01,      
    momentum=0.937,  
    weight_decay=0.001,
    warmup_epochs=3.0,   
    warmup_momentum=0.8, 
    warmup_bias_lr=0.1,
    
    box=25.0,       
    cls=1.1,       
    dfl=2.0,       
        
    # --- 데이터 증강 (Augmentation) ---
    mosaic=0.0,      
    mixup=0.0,       
    copy_paste=0.0,  

    hsv_h=0.015,     
    hsv_s=0.7,       
    hsv_v=0.4,       
    degrees=0.0,     
    translate=0.05,   
    scale=0.9,       
    shear=0.0,       
    perspective=0.0, 
    fliplr=0.5,      
    flipud=0.0,      
)



# -------- Inference (최적 F1 근처로) ----------
INFER = dict(
  STAGE2=dict(
    weights=str(MODEL_DIR / "stage2" / "train"/ "weights" / "best.pt"),
    source=str(DATA_DIR / "test" / "images"),
    conf=0.20,
    iou=0.5,
    save_txt=True,
    save_dir=str(MODEL_DIR / "infer_stage2"),

    use_tiling=True,
    tile=768,
    overlap=0.55,
    pad_ratio=0.10,

    # 🟢 새로 추가
    tta=True,  # model.predict(..., augment=True)
    per_class_conf={0: 0.35, 1: 0.55},  # 0: spalling, 1: flat
  ),
)

# --------- 데이터셋 빌드(전처리) 설정 ---------
BUILD_STAGE2 = dict(
    src_root=str(DATA_DIR),             # data/train|valid|test 사용
    out_root=str(DATA_DIR),             # *_tiles로 생성
    splits=("train", "valid", "test"),

    stage1_weights=str(MODEL_DIR / "stage1" / "weights" / "best.pt"),
    conf1=STAGE1["conf"],               # wheel 추론 conf/iou
    iou1=STAGE1["iou"],

    # 타일링/패딩 끔: 휠 박스 그대로 크롭
    use_tiling=False,
    #tile=STAGE2["tile"],
    #overlap=STAGE2["overlap"],
    #pad_ratio=0.0,

    # ROI-라벨 교집합 커버리지 임계값
    min_coverage=0.25,                  # 새 키
    min_cov=0.25,                       # 기존 코드 호환(일부 버전은 이 키를 참조)

    # 원본 라벨 id → Stage2 id
    # (원본: wheel=0, spalling=1, flat=2) → (Stage2: spalling=0, flat=1)
    class_map={1: 0, 2: 1},
)

# === 6분할 512 정사각 패딩 전용 Stage2 빌드 설정 ===
BUILD_STAGE2_512 = dict(
    src_root=str(DATA_DIR),             # data/train|valid|test 사용
    out_root=str(DATA_DIR),             # *_tiles2_512로 생성
    splits=("train", "valid"),                  # train만 타일 생성

    stage1_weights=str(MODEL_DIR / "stage1" / "weights" / "best.pt"),
    conf1=STAGE1["conf"],               # wheel 추론 conf/iou
    iou1=STAGE1["iou"],

    # ROI-라벨 교집합 커버리지 임계값
    min_coverage=0.25,
    min_cov=0.25,

    # 라벨 매핑 (원본: wheel=0, spalling=1, flat=2 → Stage2: spalling=0, flat=1)
    class_map={1: 0, 2: 1},

    # 6분할 + 512 정사각 패딩 관련 설정
    six_split=True,                     # 6분할 모드 활성화
    six_overlap=0.18,                   # 10~20% 겹침
    square_side=640,                    # 최종 정사각 크기
    pad_mode="reflect",                 # reflect | constant
    # also_default=False,               # True면 기본 타일링도 병행 (기본 False)
)



# 실행 모드: train_stage1 | build_stage2 | train_stage2 | infer_stage1 | infer_stage2 | infer_two_stage

RUN_MODE="train_stage2"   

# 두 단계 연결 추론용 입력 폴더(선택)
INFER_INPUT_DIR = str(DATA_DIR / "test" / "images")