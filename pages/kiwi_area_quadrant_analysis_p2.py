import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="P2｜品牌象限分組（全資料）", layout="wide")

# =========================
# Load df from session (uploaded in Home)
# =========================
@st.cache_data(show_spinner=False)
def load_df_from_session(data_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data_bytes))

if "data_bytes" not in st.session_state:
    st.warning("尚未在入口頁上傳資料，請先回到 Home 上傳 Excel。")
    st.stop()

df = load_df_from_session(st.session_state["data_bytes"])

# =========================
# Config (與 p1 一致：X=成長率、Y=回推平均營業額)
# =========================
X_COL = "2025成長率"
Y_COL = "2025回推平均營業額"

BRAND_COL = "品牌"
INCLUDE_COL = "帶入象限分析"  # 只取 y

# =========================
# Helpers
# =========================
def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan})

    is_pct = s2.str.contains("%", na=False)
    s2 = s2.str.replace(",", "", regex=False).str.replace("%", "", regex=False)

    out = pd.to_numeric(s2, errors="coerce")
    out = np.where(is_pct, out / 100.0, out)
    return pd.Series(out, index=s.index, dtype="float64")

# =========================
# Validate required columns
# =========================
need_cols = [X_COL, Y_COL, BRAND_COL, INCLUDE_COL]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.error(f"缺少欄位：{missing}\n目前欄位：{list(df.columns)}")
    st.stop()

# =========================
# Prepare data (全資料，不做任何細分篩選)
# =========================
df = df.copy()

# 只保留 帶入象限分析 = y
df[INCLUDE_COL] = df[INCLUDE_COL].astype(str).str.strip().str.lower()
df = df[df[INCLUDE_COL] == "y"].copy()

# 轉數值
df["_x_raw"] = to_numeric_series(df[X_COL])  # 成長率
df["_y_raw"] = to_numeric_series(df[Y_COL])  # 回推平均營業額
df = df.dropna(subset=["_x_raw", "_y_raw"]).copy()

# 成長率逐筆偵測（與 p1 同邏輯）：32.15 -> 0.3215
df["_x"] = df["_x_raw"].where(df["_x_raw"].abs() <= 1.5, df["_x_raw"] / 100.0)
df["_y"] = df["_y_raw"]

# =========================
# UI: choose cut mode
# =========================
st.title("P2｜品牌象限分組（全資料）")

cut_mode = st.radio("分界點計算方式", ["平均值", "中位數"], index=0, horizontal=True)

if cut_mode == "平均值":
    x_cut = float(df["_x"].mean())
    y_cut = float(df["_y"].mean())
else:
    x_cut = float(df["_x"].median())
    y_cut = float(df["_y"].median())

c1, c2, c3 = st.columns(3)
c1.metric("X 分界值（成長率）", f"{x_cut:.2%}")
c2.metric("Y 分界值（回推平均營業額）", f"{y_cut:,.2f}")
c3.metric("資料筆數", f"{len(df):,}")

# =========================
# Quadrant classification (用全資料分界)
# =========================
conds = [
    (df["_x"] >= x_cut) & (df["_y"] >= y_cut),
    (df["_x"] <  x_cut) & (df["_y"] >= y_cut),
    (df["_x"] <  x_cut) & (df["_y"] <  y_cut),
    (df["_x"] >= x_cut) & (df["_y"] <  y_cut),
]
q_labels = ["第一象限", "第二象限", "第三象限", "第四象限"]
df["象限"] = np.select(conds, q_labels, default="未分類")

# 象限排序鍵（第一、第二、第三、第四）
q_order_map = {"第一象限": 1, "第二象限": 2, "第三象限": 3, "第四象限": 4}
df["_q_order"] = df["象限"].map(q_order_map).fillna(99).astype(int)

# =========================
# Output: simplest table (full data + quadrant)
# 排序：品牌 -> 象限（1~4）
# =========================
st.subheader("全資料門店明細（含象限分類）")

# 若你只想先看關鍵欄位，可以在這裡縮欄位；你說「直接以完整資料補上象限欄位」，所以這裡保留全欄位
out_df = df.sort_values([BRAND_COL, "_q_order"]).drop(columns=["_q_order"])

# 顯示筆數可能很大，先給個可調整顯示上限（不影響資料本身）
max_rows = st.number_input("表格顯示筆數上限（不影響計算）", min_value=100, max_value=50000, value=2000, step=100)

st.dataframe(out_df.head(int(max_rows)), width="stretch")