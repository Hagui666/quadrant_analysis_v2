import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="象限分析", layout="wide")

@st.cache_data(show_spinner=False)
def load_df_from_session(data_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data_bytes))

if "data_bytes" not in st.session_state:
    st.warning("尚未在入口頁上傳資料，請先回到 Home 上傳 Excel。")
    st.stop()

df = load_df_from_session(st.session_state["data_bytes"])


# =========================
# Dark theme
# =========================
st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; color: #ffffff; }
      [data-testid="stSidebar"] { background-color: #0e1117; }
      h1, h2, h3, h4, h5, h6, p, div, span, label { color: #ffffff !important; }
      .note { font-size: 12px; opacity: 0.85; margin-top: -6px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Config (欄位映射)
# =========================
# X_COL = "2025回推平均營業額"
# Y_COL = "2025成長率"
X_COL = "2025成長率"
Y_COL = "2025回推平均營業額"

BRAND_COL = "品牌"
ZONE_COL = "分區編碼"
CITY_COL = "城市"
CITY_CODE_COL = "城市編碼"
CIRCLE_COL = "商圈名稱(kiwi)"

MATCHED_COL = "matched_name"
STORE_UI_COL = "門店"

INCLUDE_COL = "帶入象限分析"  # 只取 y
ADDRESS_COL = "地址"          # 你已把 地址1 改名為 地址

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

def leading_int(text: str):
    if text is None:
        return 10**12
    m = re.match(r"\s*(\d+)", str(text))
    return int(m.group(1)) if m else 10**12

def parse_optional_float(text: str):
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    try:
        return float(t.replace(",", ""))
    except Exception:
        return None

def apply_optional_range(df_in: pd.DataFrame, col_internal: str, min_v, max_v) -> pd.DataFrame:
    out = df_in
    if min_v is not None:
        out = out[out[col_internal] >= min_v]
    if max_v is not None:
        out = out[out[col_internal] <= max_v]
    return out

def multiselect_with_all_sidebar(label, options, key_prefix, default_all=True):
    """
    Sidebar 多選元件 + 全選/全不選，支援連動 options 變動（避免 default 不在 options）
    """
    options = list(options)
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    key_ms = f"{key_prefix}_ms"

    if key_ms not in st.session_state:
        st.session_state[key_ms] = options[:] if default_all else []

    # 先把舊選取值裁成 options 的子集合，避免 Streamlit 報錯
    current = st.session_state.get(key_ms, [])
    current = [v for v in current if v in options]

    if default_all and len(current) == 0 and len(options) > 0:
        current = options[:]

    st.session_state[key_ms] = current

    with c1:
        if st.button("全選", key=f"{key_prefix}_btn_all"):
            st.session_state[key_ms] = options[:]
    with c2:
        if st.button("全不選", key=f"{key_prefix}_btn_none"):
            st.session_state[key_ms] = []

    sel = st.sidebar.multiselect(
        label="",
        options=options,
        default=st.session_state[key_ms],
        key=key_ms
    )
    return sel

# =========================
# Prepare data
# =========================
df = df.copy()

# 只保留 帶入象限分析 = y
df[INCLUDE_COL] = df[INCLUDE_COL].astype(str).str.strip().str.lower()
df = df[df[INCLUDE_COL] == "y"].copy()

df[STORE_UI_COL] = df[MATCHED_COL].astype(str).replace({"nan": np.nan}).fillna("").str.strip()
df[STORE_UI_COL] = df[STORE_UI_COL].replace({"": np.nan})

# df["_x_raw"] = to_numeric_series(df[X_COL])
# df["_y_raw"] = to_numeric_series(df[Y_COL])
df["_x_raw"] = to_numeric_series(df[X_COL])  # X = 成長率
df["_y_raw"] = to_numeric_series(df[Y_COL])  # Y = 回推平均營業額
df = df.dropna(subset=["_x_raw", "_y_raw"]).copy()

# 成長率逐筆偵測（避免整欄誤除 100）
# df["_y"] = df["_y_raw"].where(df["_y_raw"].abs() <= 1.5, df["_y_raw"] / 100.0)
df["_x"] = df["_x_raw"].where(df["_x_raw"].abs() <= 1.5, df["_x_raw"] / 100.0) # 成長率逐筆偵測（套用在 X 軸）
# df["_x"] = df["_x_raw"]
df["_y"] = df["_y_raw"] # 營業額不需要百分比轉換

df["_city_order"] = df[CITY_CODE_COL].apply(leading_int)

# =========================
# Sidebar filters (全部放左側)
# =========================
st.sidebar.header("篩選器（連動）")

# (1) 品牌
all_brands = sorted(df[BRAND_COL].dropna().astype(str).unique().tolist())
brand_pick = multiselect_with_all_sidebar("品牌（多選）", all_brands, key_prefix="brand", default_all=True)

fdf = df.copy()
fdf = fdf[fdf[BRAND_COL].astype(str).isin(brand_pick)].copy() if brand_pick else fdf.iloc[0:0].copy()

# (2) 分區編碼
zone_options = sorted(fdf[ZONE_COL].dropna().astype(str).unique().tolist())
zone_pick = multiselect_with_all_sidebar("分區編碼（多選）", zone_options, key_prefix="zone", default_all=True)

fdf = fdf[fdf[ZONE_COL].astype(str).isin(zone_pick)].copy() if zone_pick else fdf.iloc[0:0].copy()

# (3) 城市（依城市編碼排序）
tmp_city = (
    fdf[[CITY_COL, "_city_order"]]
    .dropna()
    .assign(_city=lambda d: d[CITY_COL].astype(str))
    .drop_duplicates(subset=["_city"])
    .sort_values(["_city_order", "_city"])
)
city_options_sorted = tmp_city["_city"].tolist()
city_pick = multiselect_with_all_sidebar("城市（多選）", city_options_sorted, key_prefix="city", default_all=True)

fdf = fdf[fdf[CITY_COL].astype(str).isin(city_pick)].copy() if city_pick else fdf.iloc[0:0].copy()

# (4) 商圈
circle_options = sorted(fdf[CIRCLE_COL].dropna().astype(str).unique().tolist())
circle_pick = multiselect_with_all_sidebar("商圈名稱(kiwi)（多選）", circle_options, key_prefix="circle", default_all=True)

fdf = fdf[fdf[CIRCLE_COL].astype(str).isin(circle_pick)].copy() if circle_pick else fdf.iloc[0:0].copy()

# (5) 數值篩選（可空）
st.sidebar.markdown("---")
st.sidebar.markdown("**數值篩選（可選）**")
st.sidebar.markdown('<div class="note">※ 最小/最大留空代表不啟用此數值篩選</div>', unsafe_allow_html=True)

x_min = parse_optional_float(st.sidebar.text_input(f"{X_COL} 最小值", value="", key="xmin"))
x_max = parse_optional_float(st.sidebar.text_input(f"{X_COL} 最大值", value="", key="xmax"))
y_min = parse_optional_float(st.sidebar.text_input(f"{Y_COL} 最小值", value="", key="ymin"))
y_max = parse_optional_float(st.sidebar.text_input(f"{Y_COL} 最大值", value="", key="ymax"))

fdf = apply_optional_range(fdf, "_x", x_min, x_max)
fdf = apply_optional_range(fdf, "_y", y_min, y_max)

# 顯示標籤、分界模式
st.sidebar.markdown("---")
show_labels = st.sidebar.toggle("顯示資料標籤（門店）", value=False)
cut_mode = st.sidebar.radio("分界點", ["平均值", "中位數"], index=0)

if len(fdf) == 0:
    st.warning("目前篩選結果為空，請調整左側篩選條件。")
    st.stop()

# =========================
# Cuts + quadrant
# =========================
if cut_mode == "平均值":
    x_cut = float(fdf["_x"].mean())
    y_cut = float(fdf["_y"].mean())
else:
    x_cut = float(fdf["_x"].median())
    y_cut = float(fdf["_y"].median())

conds = [
    (fdf["_x"] >= x_cut) & (fdf["_y"] >= y_cut),
    (fdf["_x"] < x_cut) & (fdf["_y"] >= y_cut),
    (fdf["_x"] < x_cut) & (fdf["_y"] < y_cut),
    (fdf["_x"] >= x_cut) & (fdf["_y"] < y_cut),
]
q_labels = ["第一象限", "第二象限", "第三象限", "第四象限"]
fdf = fdf.copy()
fdf["象限"] = np.select(conds, q_labels, default="未分類")

# =========================
# Main UI
# =========================
st.title("2025 回推平均營業額 × 2025 成長率（互動）")

c1, c2, c3 = st.columns(3)
# c1.metric("X 分界值", f"{x_cut:,.2f}")
c1.metric("X 分界值", f"{x_cut:.2%}")
# c2.metric("Y 分界值", f"{y_cut:.2%}")
c2.metric("Y 分界值", f"{y_cut:,.2f}")
c3.metric("資料筆數", f"{len(fdf):,}")

hover_dict = {
    STORE_UI_COL: True,
    BRAND_COL: True,
    ZONE_COL: True,
    CITY_COL: True,
    CIRCLE_COL: True,
    X_COL: True,
    Y_COL: True,
    "象限": True,
}
if ADDRESS_COL in fdf.columns:
    hover_dict[ADDRESS_COL] = True

fig = px.scatter(
    fdf,
    x="_x",
    y="_y",
    color=BRAND_COL,
    text=STORE_UI_COL,
    hover_data=hover_dict,
    labels={"_x": X_COL, "_y": Y_COL, STORE_UI_COL: "門店"},
    title=f"散點圖（{cut_mode}分界｜顏色=品牌）"
)

# fig.add_vline(x=x_cut, line_dash="dash", annotation_text=f"X分界: {x_cut:,.2f}", annotation_position="top")
fig.add_vline(x=x_cut, line_dash="dash", annotation_text=f"X分界: {x_cut:.2%}", annotation_position="top")
# fig.add_hline(y=y_cut, line_dash="dash", annotation_text=f"Y分界: {y_cut:.2%}", annotation_position="right")
fig.add_hline(y=y_cut, line_dash="dash", annotation_text=f"Y分界: {y_cut:,.2f}", annotation_position="right")

if show_labels:
    fig.update_traces(mode="markers+text", textposition="top center")
else:
    fig.update_traces(mode="markers")

# fig.update_yaxes(tickformat=".0%")
fig.update_xaxes(tickformat=".0%")   # X 軸是成長率，用百分比顯示
fig.update_layout(template="plotly_dark", hovermode="closest", legend_title_text="品牌", height=900)

st.plotly_chart(fig, use_container_width=True)

st.subheader("各象限筆數")

# 固定四象限 + 描述欄位（不會因為當前篩選缺象限而消失）
q_meta = pd.DataFrame(
    [
        ["第一象限", "高營收, 高成長", "明星門市"],
        ["第二象限", "高營收, 低成長", "金牛門市"],
        ["第三象限", "低營收, 低成長", "問題門市"],
        ["第四象限", "低營收, 高成長", "潛力門市"],
    ],
    columns=["象限", "特徵", "店類型"]
)

q_cnt = (
    fdf["象限"]
    .value_counts()
    .rename_axis("象限")
    .reset_index(name="count")
)

q_table = (
    q_meta.merge(q_cnt, on="象限", how="left")
    .fillna({"count": 0})
)

# count 轉成整數（避免顯示 0.0）
q_table["count"] = q_table["count"].astype(int)

# 欄位排序：象限、特徵、店類型、count
q_table = q_table[["象限", "特徵", "店類型", "count"]]

st.dataframe(q_table, width="stretch")

st.subheader("篩選後門店明細")

detail_cols = [
    BRAND_COL,         # 品牌
    STORE_UI_COL,      # 門店（matched_name 改名）
    ZONE_COL,          # 分區編碼
    CITY_COL,          # 城市
    "行政區",          # 來源欄位（新版資料有）
    ADDRESS_COL,       # 地址
    CIRCLE_COL,        # 商圈名稱(kiwi)
    X_COL,             # 2025成長率
    Y_COL,             # 2025回推平均營業額
    "象限",            # 程式計算後欄位
]

# 只取存在的欄位（避免哪天欄位名變動直接報錯）
detail_cols_exist = [c for c in detail_cols if c in fdf.columns]

# 讓「門店」欄位顯示在 dataframe 上就是「門店」(已經是 STORE_UI_COL)
# 如你還想強制顯示欄名順序，以上已經處理

st.dataframe(
    fdf[detail_cols_exist].copy(),
    width="stretch"
)
