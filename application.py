# application.py
import os
import re
import cv2
import pandas as pd
from typing import List, Tuple, Optional, Dict

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from os import PathLike

# -------------------- 페이지 설정 --------------------
st.set_page_config(page_title="KORAIL Wheel Defect Detection", layout="wide")

# -------------------- Config 기본값 --------------------
try:
    from src.config import YOLO_MODEL as CFG_YOLO_MODEL
except Exception:
    CFG_YOLO_MODEL = "models/best.pt"  # 기본 경로 가정

try:
    from src.config import TEST_FOLDER as CFG_TEST_FOLDER
except Exception:
    CFG_TEST_FOLDER = "./data/test/images"

try:
    from src.config import ALLOWED_DEFECTS as CFG_ALLOWED_DEFECTS
except Exception:
    CFG_ALLOWED_DEFECTS = ['wheel', 'spalling', 'flat']

try:
    from src.config import BATCH_SIZE as CFG_BATCH_SIZE
except Exception:
    CFG_BATCH_SIZE = 3  # 화면(좌3·우3)에 맞춰 3개 재생 단위

try:
    from src.config import TICK_MS as CFG_TICK_MS
except Exception:
    CFG_TICK_MS = 2000  # UI에서 제거, 코드 상수로만 관리

IMG_SIZE = 1280
DISPLAY_W, DISPLAY_H = 200, 300    # 썸네일 내부 리사이즈
AUTO_REFRESH_KEY = "__auto_refresh__"

# ✅ 컬럼 구성: Select, Position, Wheel ID, ...
DISPLAY_COLS = ['Position', 'Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']

# -------------------- 외부 모듈 --------------------
from src.appDefect import (
    Detector,
    list_images,
    find_image_by_stem,
)

# -------------------- 스타일 --------------------
st.markdown("""
<style>
.main-title{font-size:32px;font-weight:800;color:#003087;text-align:center;margin:8px 0 16px 0;}
.camera-box{border:1px solid #e5e7eb;border-radius:8px;background:#fff;height:300px;width:100%;}
.subtle{color:#6b7280;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">KORAIL Wheel Defect Detection System</div>', unsafe_allow_html=True)

# -------------------- 세션 기본값 --------------------
defaults = dict(
    detector=None,
    model_path="",
    conf_threshold=0.25,

    images_all=[],
    pairs=[],
    left_paths=[],
    right_paths=[],

    preds_cache={},                 # 표시용(기존)
    preds_cache_raw={},             # 원본 rows 캐시(추가)

    cursor=0,
    cursorL=0,
    cursorR=0,
    is_running=False,
    preloaded=False,

    # 누적 테이블
    defects_table=None,             # flat/spalling (wheel 내부일 때만), 이미지당 1행
    wheels_table=None,              # wheel만 있고 flat/spalling이 전혀 없는 이미지

    batch_size=int(CFG_BATCH_SIZE),
    tick_ms=int(CFG_TICK_MS),
    test_folder="",

    # 선택 상태
    active_source=None,             # 'defects' | 'wheels' | None
    defects_select=None,
    wheels_select=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- 유틸 --------------------
def resolve_abs_path(p: str) -> str:
    if p is None:
        p = ""
    if isinstance(p, (PathLike,)):
        p = os.fspath(p)
    p = os.path.expanduser((str(p) or "").strip())
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(os.getcwd(), p))

def rows_to_display_df(rows: list) -> pd.DataFrame:
    """
    참고: 메인 테이블은 아래 _aggregate_for_path()를 통해 다시 구성되므로,
    여기서는 원본 rows를 단순 변환. (Position 컬럼은 집계 단계에서 추가)
    """
    base_cols = ['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']
    if not rows:
        return pd.DataFrame(columns=base_cols)
    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.rename(columns={
        'wheel_id': 'Wheel ID',
        'defect_type': 'Defect Type',
        'confidence': 'Confidence',
        'area_ratio': 'Area Ratio',
        'time': 'Time',
    })
    for c in base_cols:
        if c not in df.columns:
            df[c] = "" if c in ('Wheel ID','Defect Type','Time') else 0.0
    df = df[base_cols]
    df['Wheel ID'] = df['Wheel ID'].astype(str)
    df['Defect Type'] = df['Defect Type'].astype(str)
    df['Time'] = df['Time'].astype(str)
    df[['Confidence','Area Ratio']] = df[['Confidence','Area Ratio']].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def cached_read_resized(abs_path: str, w: int = DISPLAY_W, h: int = DISPLAY_H):
    img = cv2.imread(abs_path)
    if img is None:
        return None
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def make_lr_pairs(paths: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    L, R = {}, {}
    for p in paths:
        name = os.path.basename(p)
        stem, _ = os.path.splitext(name)
        s = (stem or "").strip()
        if not s:
            continue
        if s[0] in ('L','l'):
            key = (s[1:] or s).lower()
            L[key] = p
        elif s[0] in ('R','r'):
            key = (s[1:] or s).lower()
            R[key] = p
        else:
            L[s.lower()] = p
    keys = sorted(set(L) | set(R))
    return [(L.get(k), R.get(k)) for k in keys]

def stem_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def parse_position_and_id(filename: str):
    """파일명에서 Position(Left/Right)과 4자리 Wheel ID를 추출"""
    name = os.path.basename(filename)
    stem, _ = os.path.splitext(name)
    if not stem:
        return "", ""
    pos = ""
    first_char = stem[0].upper()
    if first_char == "L":
        pos = "Left"
    elif first_char == "R":
        pos = "Right"
    m = re.search(r"[LRlr](\d{4})", stem)
    wheel_num = m.group(1) if m else stem
    return pos, wheel_num

# -------------------- 사이드바 --------------------
with st.sidebar:
    st.header("Settings")
    in_model = st.text_input("YOLO model path", value=CFG_YOLO_MODEL)
    in_folder = st.text_input("Test folder", value=CFG_TEST_FOLDER)
    in_conf = st.slider("Confidence", 0.0, 1.0, st.session_state.conf_threshold, 0.01)

    colA, colB = st.columns(2)
    with colA:
        btn_run = st.button("Run Detection", type="primary")
    with colB:
        btn_reset = st.button("Reset")

st.session_state.conf_threshold = float(in_conf)

if btn_reset:
    for k in list(st.session_state.keys()):
        if k in defaults:
            st.session_state[k] = defaults[k]
    st.success("State reset.")

# -------------------- 모델/데이터 지연 로딩 --------------------
def ensure_loaded(model_path_input: str, folder_input: str) -> bool:
    mp = resolve_abs_path(model_path_input)
    dp = resolve_abs_path(folder_input)

    if not os.path.exists(mp):
        st.error(f"모델 파일이 없습니다: {mp}")
        return False
    if not os.path.isdir(dp):
        st.error(f"테스트 폴더가 없습니다: {dp}")
        return False

    detector = None
    try:
        try:
            detector = Detector(mp, CFG_ALLOWED_DEFECTS, imgsz=IMG_SIZE, wheel_classes=["wheel"])
        except TypeError:
            detector = Detector(mp, CFG_ALLOWED_DEFECTS)
        try:
            if hasattr(detector, "warmup") and callable(detector.warmup):
                detector.warmup(st.session_state.conf_threshold)
        except Exception as we:
            st.warning(f"Warmup skipped: {we}")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return False

    paths = list_images(dp)
    if not paths:
        st.warning(f"테스트 폴더에 이미지가 없습니다: {dp}")
        return False
    pairs = make_lr_pairs(paths)

    left_paths, right_paths = [], []
    for Lp, Rp in pairs:
        if Lp: left_paths.append(Lp)
        if Rp: right_paths.append(Rp)

    st.info("이미지 로드 및 예측 캐싱중…")
    progress = st.progress(0)
    cache_disp: Dict[str, pd.DataFrame] = {}
    cache_raw: Dict[str, pd.DataFrame] = {}
    flat = [*left_paths, *right_paths]
    total = len(flat)

    for i, p in enumerate(flat):
        _ = cached_read_resized(p)
        if p not in cache_disp:
            try:
                rows = detector.predict_rows(p, st.session_state.conf_threshold)
                disp_df = rows_to_display_df(rows)
                raw_df = pd.DataFrame([r.__dict__ for r in rows]) if rows else pd.DataFrame()
                cache_disp[p] = disp_df
                cache_raw[p] = raw_df
            except Exception as pe:
                st.warning(f"예측 실패: {os.path.basename(p)} - {pe}")
                cache_disp[p] = pd.DataFrame(columns=DISPLAY_COLS)
                cache_raw[p] = pd.DataFrame(columns=['wheel_id','defect_type','confidence','area_ratio','time'])
        progress.progress((i + 1) / max(1, total))
    progress.empty()
    st.success("이미지 로드 및 예측 캐싱 완료!")

    st.session_state.detector = detector
    st.session_state.model_path = mp
    st.session_state.images_all = paths
    st.session_state.pairs = pairs
    st.session_state.left_paths = left_paths
    st.session_state.right_paths = right_paths
    st.session_state.preds_cache = cache_disp
    st.session_state.preds_cache_raw = cache_raw
    st.session_state.cursor = 0
    st.session_state.cursorL = 0
    st.session_state.cursorR = 0
    st.session_state.preloaded = True
    st.session_state.is_running = True
    st.session_state.test_folder = dp

    # 누적 테이블 초기화
    st.session_state.defects_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])
    st.session_state.wheels_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])

    # 선택 상태 초기화
    st.session_state.active_source = None
    st.session_state.defects_select = None
    st.session_state.wheels_select = None

    return True

if btn_run:
    ensure_loaded(in_model, in_folder)

# -------------------- 집계 로직 --------------------
TARGET_DEFECTS = {'flat', 'spalling'}

def _aggregate_for_path(p: str):
    """
    이미지 단위 집계 (image_key = 파일명 stem)
    - defect list: wheel 존재 & TARGET_DEFECTS가 1개 이상 존재 → 1행으로 합침(타입은 'flat,spalling' 식으로 병합)
    - wheel list: wheel 존재 & TARGET_DEFECTS가 하나도 없음 → 1행
    + Position(Left/Right)과 Wheel ID(4자리) 반영
    """
    raw = st.session_state.preds_cache_raw.get(p, pd.DataFrame())
    if raw is None or raw.empty:
        return None, None

    # 결측 방어
    for col in ['wheel_id','defect_type','confidence','area_ratio','time']:
        if col not in raw.columns:
            raw[col] = None

    image_key = stem_from_path(p)  # 파일명 stem (검색용)
    pos, wheel_short_id = parse_position_and_id(p)

    types = raw['defect_type'].astype(str).str.lower().fillna('')
    has_wheel = (types == 'wheel').any()
    present_targets = set(types[types.isin(TARGET_DEFECTS)].tolist())

    defect_row = None
    wheel_row = None

    if has_wheel and present_targets:
        sub = raw[raw['defect_type'].astype(str).str.lower().isin(TARGET_DEFECTS)].copy()
        joined_types = ",".join(sorted(present_targets))  # 예: "flat,spalling"
        conf = float(sub['confidence'].max(skipna=True) if 'confidence' in sub.columns else 0.0)
        area = float(sub['area_ratio'].max(skipna=True) if 'area_ratio' in sub.columns else 0.0)
        time_val = str(sub['time'].astype(str).fillna('').iloc[0] if 'time' in sub.columns and not sub.empty else "")
        defect_row = {
            "Select": False,
            "Position": pos,
            "Wheel ID": wheel_short_id,
            "Defect Type": joined_types,
            "Confidence": conf,
            "Area Ratio": area,
            "Time": time_val,
            # 내부 검색용 숨은 키를 유지하고 싶다면 별도 보관 가능
            # "_image_key": image_key,
        }

    # wheel-only: 대상 결함이 하나도 없을 때만
    if has_wheel and not present_targets:
        subw = raw[raw['defect_type'].astype(str).str.lower() == 'wheel'].copy()
        conf_w = float(subw['confidence'].max(skipna=True) if 'confidence' in subw.columns else 0.0)
        area_w = float(subw['area_ratio'].max(skipna=True) if 'area_ratio' in subw.columns else 0.0)
        time_w = str(subw['time'].astype(str).fillna('').iloc[0] if 'time' in subw.columns and not subw.empty else "")
        wheel_row = {
            "Select": False,
            "Position": pos,
            "Wheel ID": wheel_short_id,
            "Defect Type": "wheel",
            "Confidence": conf_w,
            "Area Ratio": area_w,
            "Time": time_w,
            # "_image_key": image_key,
        }

    return defect_row, wheel_row

def _upsert_by_image_key(df: pd.DataFrame, row: Optional[dict]) -> pd.DataFrame:
    """Wheel ID(숫자 4자리)를 키로 upsert. (동일 이미지 기준 중복 방지)"""
    if row is None:
        return df
    if df is None or df.empty:
        return pd.DataFrame([row])
    mask = (df["Wheel ID"].astype(str) == str(row["Wheel ID"]))
    if mask.any():
        idx = df.index[mask][0]
        for k, v in row.items():
            df.at[idx, k] = v
        df.at[idx, "Select"] = False
        return df
    else:
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# -------------------- 진행/오토리프레시 --------------------
def advance_once():
    if not st.session_state.is_running:
        return ([], []), True

    L = st.session_state.left_paths or []
    R = st.session_state.right_paths or []
    cL = st.session_state.get("cursorL", 0)
    cR = st.session_state.get("cursorR", 0)

    take = 3
    left_batch  = L[cL:cL+take]
    right_batch = R[cR:cR+take]

    st.session_state.cursorL = cL + len(left_batch)
    st.session_state.cursorR = cR + len(right_batch)

    if st.session_state.defects_table is None:
        st.session_state.defects_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])
    if st.session_state.wheels_table is None:
        st.session_state.wheels_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])

    for p in [*left_batch, *right_batch]:
        defect_row, wheel_row = _aggregate_for_path(p)
        st.session_state.defects_table = _upsert_by_image_key(st.session_state.defects_table, defect_row)
        st.session_state.wheels_table = _upsert_by_image_key(st.session_state.wheels_table, wheel_row)

    finished = (st.session_state.cursorL >= len(L)) and (st.session_state.cursorR >= len(R))
    if finished:
        st.session_state.is_running = False
    return (left_batch, right_batch), finished

if st.session_state.is_running and st.session_state.preloaded:
    st_autorefresh(interval=st.session_state.tick_ms, key=AUTO_REFRESH_KEY)

# -------------------- 실시간 L/R(각 3장) --------------------
with st.container(border=True):
    left_col, right_col = st.columns(2)

    if st.session_state.is_running:
        (left_batch, right_batch), finished = advance_once()
    else:
        (left_batch, right_batch), finished = ([], []), (
            (st.session_state.get("cursorL", 0) >= len(st.session_state.get("left_paths", [])))
            and (st.session_state.get("cursorR", 0) >= len(st.session_state.get("right_paths", [])))
        )

    COLS = 3

    with left_col:
        st.markdown("### Left Wheel Views")
        cols = st.columns(COLS)
        for i, ct in enumerate(cols):
            p = left_batch[i] if i < len(left_batch) else None
            with ct:
                if p:
                    img = cached_read_resized(p)
                    if img is not None:
                        st.image(img, width="stretch")
                    else:
                        st.markdown('<div class="camera-box"></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="camera-box"></div>', unsafe_allow_html=True)

    with right_col:
        st.markdown("### Right Wheel Views")
        cols = st.columns(COLS)
        for i, ct in enumerate(cols):
            p = right_batch[i] if i < len(right_batch) else None
            with ct:
                if p:
                    img = cached_read_resized(p)
                    if img is not None:
                        st.image(img, width="stretch")
                    else:
                        st.markdown('<div class="camera-box"></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="camera-box"></div>', unsafe_allow_html=True)

    if finished and st.session_state.preloaded:
        st.success("✅ 모든 이미지 재생 완료!")
        if st.button("🔁 다시 재생"):
            st.session_state.cursorL = 0
            st.session_state.cursorR = 0
            st.session_state.is_running = True

st.markdown("---")

# -------------------- 결과 테이블 + 요약 --------------------
st.markdown("### Inspection Result")
left, right = st.columns(2)

def _ensure_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        base = pd.DataFrame(columns=["Select", *DISPLAY_COLS])
        return base
    for c in ["Select", *DISPLAY_COLS]:
        if c not in df.columns:
            df[c] = False if c == "Select" else ("" if c in ('Position','Wheel ID','Defect Type','Time') else 0.0)
    df["Select"] = df["Select"].astype(bool)
    df["Position"] = df["Position"].astype(str)
    df["Wheel ID"] = df["Wheel ID"].astype(str)
    df["Defect Type"] = df["Defect Type"].astype(str)
    df["Time"] = df["Time"].astype(str)
    df[["Confidence","Area Ratio"]] = df[["Confidence","Area Ratio"]].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df[["Select", *DISPLAY_COLS]].copy()

with left:
    # --- 결함 리스트 ---
    with st.container(border=True):
        df_def = _ensure_table_columns(st.session_state.get("defects_table"))
        flat_count = int(df_def["Defect Type"].str.contains(r"\bflat\b", case=False, na=False).sum())
        spalling_count = int(df_def["Defect Type"].str.contains(r"\bspalling\b", case=False, na=False).sum())
        st.subheader(f"Defect List (flat {flat_count}개, spalling {spalling_count}개)")

        # 방어: 혹시라도 wheel만 들어왔으면 필터
        if not df_def.empty:
            df_def = df_def[
                df_def["Defect Type"].str.contains("flat", case=False) |
                df_def["Defect Type"].str.contains("spalling", case=False)
            ]
        if df_def.empty:
            st.write("No defects detected yet.")
            st.session_state.defects_select = df_def.copy()
        else:
            edited_def = st.data_editor(
                df_def,
                hide_index=True,
                height=360,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False, help="우측 요약 보기"),
                    "Position": st.column_config.TextColumn("Position", disabled=True),
                    "Wheel ID": st.column_config.TextColumn("Wheel ID", disabled=True),
                    "Defect Type": st.column_config.TextColumn("Defect Type", disabled=True),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f", disabled=True),
                    "Area Ratio": st.column_config.NumberColumn("Area Ratio", format="%.2f", disabled=True),
                    "Time": st.column_config.TextColumn("Time", disabled=True),
                },
                key="defects_editor"
            )
            if "Select" in edited_def.columns and edited_def["Select"].sum() > 1:
                first_idx = edited_def.index[edited_def["Select"]].tolist()[0]
                edited_def.loc[edited_def.index != first_idx, "Select"] = False
            st.session_state.defects_select = edited_def

    # --- 휠 리스트 ---
    with st.container(border=True):
        df_wheels = _ensure_table_columns(st.session_state.get("wheels_table"))
        wheel_only_count = int(len(df_wheels))
        st.subheader(f"Normal Status ({wheel_only_count}개)")

        # 방어: 결함문자 포함된 행 제거(이론상 없음)
        if not df_wheels.empty:
            mask_has_def = (
                df_wheels["Defect Type"].str.contains("flat", case=False) |
                df_wheels["Defect Type"].str.contains("spalling", case=False)
            )
            df_wheels = df_wheels[~mask_has_def]

        if df_wheels.empty:
            st.write("No wheels (without defects) detected yet.")
            st.session_state.wheels_select = df_wheels.copy()
        else:
            edited_wheels = st.data_editor(
                df_wheels,
                hide_index=True,
                height=280,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False, help="우측 요약 보기"),
                    "Position": st.column_config.TextColumn("Position", disabled=True),
                    "Wheel ID": st.column_config.TextColumn("Wheel ID", disabled=True),
                    "Defect Type": st.column_config.TextColumn("Defect Type", disabled=True),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f", disabled=True),
                    "Area Ratio": st.column_config.NumberColumn("Area Ratio", format="%.2f", disabled=True),
                    "Time": st.column_config.TextColumn("Time", disabled=True),
                },
                key="wheels_editor"
            )
            if "Select" in edited_wheels.columns and edited_wheels["Select"].sum() > 1:
                first_idx = edited_wheels.index[edited_wheels["Select"]].tolist()[0]
                edited_wheels.loc[edited_wheels.index != first_idx, "Select"] = False
            st.session_state.wheels_select = edited_wheels

# --- 상호배타 선택 ---
def _enforce_mutual_exclusive_selection():
    sel_def = st.session_state.get("defects_select")
    sel_whe = st.session_state.get("wheels_select")

    def_count = int(sel_def["Select"].sum()) if (isinstance(sel_def, pd.DataFrame) and "Select" in sel_def.columns) else 0
    whe_count = int(sel_whe["Select"].sum()) if (isinstance(sel_whe, pd.DataFrame) and "Select" in sel_whe.columns) else 0

    active_source = None
    if def_count > 0:
        active_source = 'defects'
        if whe_count > 0:
            st.session_state.wheels_select.loc[:, "Select"] = False
    elif whe_count > 0:
        active_source = 'wheels'
        if def_count > 0:
            st.session_state.defects_select.loc[:, "Select"] = False
    else:
        active_source = None

    st.session_state.active_source = active_source

_enforce_mutual_exclusive_selection()

# --- 우측: 선택 요약/시각화 ---
with right:
    with st.container(border=True):
        st.subheader("Selected Item Summary")
        source = st.session_state.get("active_source", None)
        raw_folder = st.session_state.get("test_folder", CFG_TEST_FOLDER)
        folder_to_use = resolve_abs_path(raw_folder)

        if source is None:
            st.info("왼쪽의 결함 리스트 또는 휠 리스트에서 항목을 하나 선택하세요.")
        else:
            sel_df = st.session_state.get("defects_select" if source == 'defects' else "wheels_select", pd.DataFrame())
            if sel_df is None or sel_df.empty or "Select" not in sel_df.columns or sel_df["Select"].sum() == 0:
                st.info("선택된 항목이 없습니다.")
            else:
                row = sel_df[sel_df["Select"]].drop(columns=["Select"]).iloc[0]
                wheel_id_disp = str(row.get("Wheel ID", "") or "")  # 4자리 숫자
                defect_type = str(row.get("Defect Type", "") or "")
                conf = float(row.get("Confidence", 0.0) or 0.0)
                area = float(row.get("Area Ratio", 0.0) or 0.0)
                pos = str(row.get("Position", "") or "")

                # 원본 이미지는 'L####' 또는 'R####...'로 시작하므로
                # find_image_by_stem(folder, '####') 만으로도 startswith 매치 가능
                img_col, info_col = st.columns([2,1])
                with img_col:
                    path = find_image_by_stem(folder_to_use, pos, wheel_id_disp)
                    if path:
                        det = st.session_state.detector
                        try:
                            plotted = det.plot_with_boxes(path, st.session_state.conf_threshold) if det else cv2.imread(path)
                        except Exception:
                            plotted = cv2.imread(path)
                        if plotted is not None:
                            plotted = cv2.resize(plotted, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
                            plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                            st.image(plotted, width="stretch")
                        else:
                            st.warning("⚠️ Unable to visualize detection")
                    else:
                        st.warning(f"⚠️ Image not found for ID {wheel_id_disp}")
                with info_col:
                    st.markdown(f"**Source:** {('Defect List' if source=='defects' else 'Wheel List')}")
                    st.markdown(f"**Position:** {pos}")
                    st.markdown(f"**Wheel ID:** {wheel_id_disp}")
                    st.markdown(f"**Defect Type:** {defect_type}")
                    st.markdown(f"**Confidence:** {conf:.2f}")
                    st.markdown(f"**Area Ratio:** {area:.2f}")