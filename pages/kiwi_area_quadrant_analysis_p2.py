import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="P2｜品牌象限分組", layout="wide")

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
X_COL = "2025成長率"           # X = 成長率
Y_COL = "2025回推平均營業額"   # Y = 回推平均營業額

BRAND_COL = "品牌"
ZONE_COL = "分區編碼"
CITY_COL = "城市"
CIRCLE_COL = "商圈名稱(kiwi)"
MATCHED_COL = "matched_name"
STORE_UI_COL = "門店"

INCLUDE_COL = "帶入象限分析"
ADDRESS_COL = "地址"

# 你要新增的欄位：本/競品（實際欄名不確定，先做候選）
COMP_COL_CANDIDATES = ["本/競品", "本競品", "本品/競品", "本品競品", "類別", "類型"]

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

def find_first_existing_col(df_: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df_.columns:
            return c
    return None

# =========================
# Validate required columns
# =========================
need_cols = [X_COL, Y_COL, BRAND_COL, INCLUDE_COL, MATCHED_COL]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.error(f"缺少欄位：{missing}\n目前欄位：{list(df.columns)}")
    st.stop()

# =========================
# Prepare data
# =========================
df = df.copy()

# 只保留 帶入象限分析 = y
df[INCLUDE_COL] = df[INCLUDE_COL].astype(str).str.strip().str.lower()
df = df[df[INCLUDE_COL] == "y"].copy()

# 門店欄（matched_name → 門店）
df[STORE_UI_COL] = (
    df[MATCHED_COL].astype(str)
    .replace({"nan": np.nan})
    .fillna("")
    .str.strip()
    .replace({"": np.nan})
)

# 轉數值
df["_x_raw"] = to_numeric_series(df[X_COL])  # 成長率
df["_y_raw"] = to_numeric_series(df[Y_COL])  # 回推平均營業額
df = df.dropna(subset=["_x_raw", "_y_raw"]).copy()

# 成長率逐筆偵測：32.15 -> 0.3215（與 p1 一致）
df["_x"] = df["_x_raw"].where(df["_x_raw"].abs() <= 1.5, df["_x_raw"] / 100.0)
df["_y"] = df["_y_raw"]

# =========================
# UI
# =========================
st.title("P2｜品牌象限分組（不做分區/城市等細分）")

# ✅ P2 只做品牌篩選（不做城市/分區/商圈）
brands = sorted(df[BRAND_COL].dropna().astype(str).unique().tolist())
brand_pick = st.multiselect("品牌（多選）", options=brands, default=brands)

fdf = df[df[BRAND_COL].astype(str).isin(brand_pick)].copy() if brand_pick else df.iloc[0:0].copy()
if len(fdf) == 0:
    st.warning("目前品牌篩選結果為空，請調整選擇。")
    st.stop()

cut_mode = st.radio("分界點計算方式", ["平均值", "中位數"], index=0, horizontal=True)

# ✅ 分界用「品牌篩選後」資料計算（跟 p1 動態一致）
if cut_mode == "平均值":
    x_cut = float(fdf["_x"].mean())
    y_cut = float(fdf["_y"].mean())
else:
    x_cut = float(fdf["_x"].median())
    y_cut = float(fdf["_y"].median())

c1, c2, c3 = st.columns(3)
c1.metric("X 分界值（成長率）", f"{x_cut:.2%}")
c2.metric("Y 分界值（回推平均營業額）", f"{y_cut:,.2f}")
c3.metric("資料筆數", f"{len(fdf):,}")

# =========================
# Quadrant classification (與 p1 同軸向)
# =========================
conds = [
    (fdf["_x"] >= x_cut) & (fdf["_y"] >= y_cut),
    (fdf["_x"] <  x_cut) & (fdf["_y"] >= y_cut),
    (fdf["_x"] <  x_cut) & (fdf["_y"] <  y_cut),
    (fdf["_x"] >= x_cut) & (fdf["_y"] <  y_cut),
]
q_labels = ["第一象限", "第二象限", "第三象限", "第四象限"]
fdf["象限"] = np.select(conds, q_labels, default="未分類")

q_order_map = {"第一象限": 1, "第二象限": 2, "第三象限": 3, "第四象限": 4}
fdf["_q_order"] = fdf["象限"].map(q_order_map).fillna(99).astype(int)

# =========================
# Output table (欄位跟 p1 明細一致 + 本/競品放第一欄)
# =========================
comp_col = find_first_existing_col(fdf, COMP_COL_CANDIDATES)

detail_cols = []
if comp_col is not None:
    detail_cols.append(comp_col)      # 本/競品（第一欄）

detail_cols += [
    BRAND_COL,
    STORE_UI_COL,
    ZONE_COL,
    CITY_COL,
    "行政區",
    ADDRESS_COL,
    CIRCLE_COL,
    Y_COL,     # 回推平均營業額（Y）
    X_COL,     # 成長率（X）—顯示原欄位（可能是 0.32 或 32，保留原始）
    "象限",
]

detail_cols_exist = [c for c in detail_cols if c in fdf.columns]

out_df = (
    fdf.sort_values([BRAND_COL, "_q_order"])
       .loc[:, detail_cols_exist]
       .copy()
)

st.subheader("品牌分組後門店明細（含象限）")
st.dataframe(out_df, width="stretch")