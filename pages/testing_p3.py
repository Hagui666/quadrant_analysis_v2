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
    unsafe_allow_html=True,
)

# =========================
# Config (欄位映射)
# =========================
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
ADDRESS_COL = "地址"
DISTRICT_COL = "行政區"

BRAND_X_CUT_COL = "品牌內X分界"
BRAND_Y_CUT_COL = "品牌內Y分界"
CALC_INCLUDED_COL = "納入分界計算"
CALC_SAMPLE_N_COL = "品牌分界計算樣本數"


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


def parse_optional_rate(text: str):
    """
    成長率欄位專用：
    - 允許輸入 2.871（代表 287.1%）
    - 也允許輸入 287.1%（自動轉成 2.871）
    """
    if text is None:
        return None
    t = str(text).strip().replace(",", "")
    if t == "":
        return None
    try:
        if t.endswith("%"):
            return float(t[:-1]) / 100.0
        return float(t)
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
    options = list(options)
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    key_ms = f"{key_prefix}_ms"

    if key_ms not in st.session_state:
        st.session_state[key_ms] = options[:] if default_all else []

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
        key=key_ms,
    )
    return sel


@st.cache_data(show_spinner=False)
def prepare_base_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out[INCLUDE_COL] = out[INCLUDE_COL].astype(str).str.strip().str.lower()
    out = out[out[INCLUDE_COL] == "y"].copy()

    out[STORE_UI_COL] = out[MATCHED_COL].astype(str).replace({"nan": np.nan}).fillna("").str.strip()
    out[STORE_UI_COL] = out[STORE_UI_COL].replace({"": np.nan})

    out["_x_raw"] = to_numeric_series(out[X_COL])
    out["_y_raw"] = to_numeric_series(out[Y_COL])
    out = out.dropna(subset=["_x_raw", "_y_raw"]).copy()

    # Excel 的百分比數值若已經是數值型，通常已是小數表示法
    out["_x"] = out["_x_raw"]
    out["_y"] = out["_y_raw"]
    out["_city_order"] = out[CITY_CODE_COL].apply(leading_int)
    return out


@st.cache_data(show_spinner=False)
def build_brand_cutoffs_and_classification(
    base_df: pd.DataFrame,
    cut_mode: str,
    x_min,
    x_max,
    y_min,
    y_max,
):
    calc_df = base_df.copy()
    calc_df = apply_optional_range(calc_df, "_x", x_min, x_max)
    calc_df = apply_optional_range(calc_df, "_y", y_min, y_max)

    if cut_mode == "平均值":
        brand_cut_df = (
            calc_df.groupby(BRAND_COL, dropna=False)
            .agg(
                x_cut=("_x", "mean"),
                y_cut=("_y", "mean"),
                calc_sample_n=("_x", "size"),
            )
            .reset_index()
        )
    else:
        brand_cut_df = (
            calc_df.groupby(BRAND_COL, dropna=False)
            .agg(
                x_cut=("_x", "median"),
                y_cut=("_y", "median"),
                calc_sample_n=("_x", "size"),
            )
            .reset_index()
        )

    classified_df = base_df.merge(brand_cut_df, on=BRAND_COL, how="left")
    classified_df[CALC_INCLUDED_COL] = classified_df.index.isin(calc_df.index)
    classified_df[CALC_SAMPLE_N_COL] = classified_df["calc_sample_n"].fillna(0).astype(int)

    valid_cut_mask = classified_df["x_cut"].notna() & classified_df["y_cut"].notna()
    conds = [
        valid_cut_mask & (classified_df["_x"] >= classified_df["x_cut"]) & (classified_df["_y"] >= classified_df["y_cut"]),
        valid_cut_mask & (classified_df["_x"] < classified_df["x_cut"]) & (classified_df["_y"] >= classified_df["y_cut"]),
        valid_cut_mask & (classified_df["_x"] < classified_df["x_cut"]) & (classified_df["_y"] < classified_df["y_cut"]),
        valid_cut_mask & (classified_df["_x"] >= classified_df["x_cut"]) & (classified_df["_y"] < classified_df["y_cut"]),
    ]
    q_labels = ["第一象限", "第二象限", "第三象限", "第四象限"]
    classified_df["象限"] = np.select(conds, q_labels, default="未分類")
    classified_df[BRAND_X_CUT_COL] = classified_df["x_cut"]
    classified_df[BRAND_Y_CUT_COL] = classified_df["y_cut"]

    return classified_df, brand_cut_df, calc_df


# =========================
# Prepare data
# =========================
base_df = prepare_base_dataframe(df)
if len(base_df) == 0:
    st.warning("目前沒有可用於象限分析的資料。")
    st.stop()


# =========================
# Sidebar filters
# =========================
st.sidebar.header("篩選器")

st.sidebar.subheader("A. 計算方式設定")
cut_mode = st.sidebar.radio("品牌內分界的計算方式", ["平均值", "中位數"], index=0)
st.sidebar.markdown(
    '<div class="note">數值篩選只影響「品牌內 X/Y 分界」的計算，不會直接刪除圖上的點；可用來排除極端值。</div>',
    unsafe_allow_html=True,
)

x_min_calc = parse_optional_rate(st.sidebar.text_input(f"{X_COL} 最小值（計算用）", value="", key="calc_xmin"))
x_max_calc = parse_optional_rate(st.sidebar.text_input(f"{X_COL} 最大值（計算用）", value="", key="calc_xmax"))
y_min_calc = parse_optional_float(st.sidebar.text_input(f"{Y_COL} 最小值（計算用）", value="", key="calc_ymin"))
y_max_calc = parse_optional_float(st.sidebar.text_input(f"{Y_COL} 最大值（計算用）", value="", key="calc_ymax"))

classified_df, brand_cut_df, calc_df = build_brand_cutoffs_and_classification(
    base_df=base_df,
    cut_mode=cut_mode,
    x_min=x_min_calc,
    x_max=x_max_calc,
    y_min=y_min_calc,
    y_max=y_max_calc,
)

st.sidebar.markdown("---")
st.sidebar.subheader("B. 呈現篩選")
st.sidebar.markdown(
    '<div class="note">以下篩選只控制畫面顯示，不會重新計算品牌內分界與象限。</div>',
    unsafe_allow_html=True,
)

all_brands = sorted(classified_df[BRAND_COL].dropna().astype(str).unique().tolist())
brand_pick = multiselect_with_all_sidebar("品牌（多選）", all_brands, key_prefix="display_brand", default_all=True)

all_zones = sorted(classified_df[ZONE_COL].dropna().astype(str).unique().tolist())
zone_pick = multiselect_with_all_sidebar("分區編碼（多選）", all_zones, key_prefix="display_zone", default_all=True)

tmp_city = (
    classified_df[[CITY_COL, "_city_order"]]
    .dropna()
    .assign(_city=lambda d: d[CITY_COL].astype(str))
    .drop_duplicates(subset=["_city"])
    .sort_values(["_city_order", "_city"])
)
city_options_sorted = tmp_city["_city"].tolist()
city_pick = multiselect_with_all_sidebar("城市（多選）", city_options_sorted, key_prefix="display_city", default_all=True)

all_circles = sorted(classified_df[CIRCLE_COL].dropna().astype(str).unique().tolist())
circle_pick = multiselect_with_all_sidebar("商圈名稱(kiwi)（多選）", all_circles, key_prefix="display_circle", default_all=True)

st.sidebar.markdown("---")
st.sidebar.subheader("圖表顯示設定")
show_labels = st.sidebar.toggle("顯示資料標籤（門店）", value=False)
marker_size = st.sidebar.slider("點大小", min_value=4, max_value=30, value=12, step=1)
label_font_size = st.sidebar.slider("資料標籤字體大小", min_value=8, max_value=28, value=14, step=1)

declutter_labels = False
x_bin_pct = 1.0
y_bin_pct = 1.5
if show_labels:
    declutter_labels = st.sidebar.toggle("避免標籤重疊（建議開啟）", value=True)
    with st.sidebar.expander("標籤防重疊強度", expanded=False):
        st.markdown(
            '<div class="note">以目前畫面顯示資料的 X/Y 軸範圍切網格，同一格只保留 1 個標籤。數字越大，保留的標籤越少。</div>',
            unsafe_allow_html=True,
        )
        x_bin_pct = st.slider("X 方向網格大小（占 X 範圍 %）", min_value=0.2, max_value=5.0, value=1.0, step=0.2)
        y_bin_pct = st.slider("Y 方向網格大小（占 Y 範圍 %）", min_value=0.2, max_value=8.0, value=1.5, step=0.2)


# =========================
# Display filters apply here only
# =========================
display_df = classified_df.copy()
if brand_pick:
    display_df = display_df[display_df[BRAND_COL].astype(str).isin(brand_pick)].copy()
else:
    display_df = display_df.iloc[0:0].copy()

if zone_pick:
    display_df = display_df[display_df[ZONE_COL].astype(str).isin(zone_pick)].copy()
else:
    display_df = display_df.iloc[0:0].copy()

if city_pick:
    display_df = display_df[display_df[CITY_COL].astype(str).isin(city_pick)].copy()
else:
    display_df = display_df.iloc[0:0].copy()

if circle_pick:
    display_df = display_df[display_df[CIRCLE_COL].astype(str).isin(circle_pick)].copy()
else:
    display_df = display_df.iloc[0:0].copy()

if len(display_df) == 0:
    st.warning("目前呈現篩選結果為空，請調整左側條件。")
    st.stop()


# =========================
# Main UI
# =========================
st.title("2025 成長率 × 2025 回推平均營業額（品牌內分界預先計算）")
st.caption(
    "目前邏輯：先依『品牌』使用全資料（僅受計算方式設定影響）預先計算各品牌的 X/Y 分界與象限；"
    "之後品牌、分區編碼、城市、商圈名稱(kiwi) 只控制畫面顯示，散點圖會依顯示結果自動縮放。"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("分界計算方式", cut_mode)
c2.metric("分界計算樣本筆數", f"{len(calc_df):,}")
c3.metric("目前顯示筆數", f"{len(display_df):,}")
c4.metric("目前顯示品牌數", f"{display_df[BRAND_COL].nunique():,}")

missing_cut_brands = sorted(
    display_df.loc[
        display_df[BRAND_X_CUT_COL].isna() | display_df[BRAND_Y_CUT_COL].isna(),
        BRAND_COL,
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
if missing_cut_brands:
    st.warning(
        "以下品牌在目前『計算方式設定』下沒有可用樣本，因此無法計算品牌內分界，相關門店會顯示為未分類："
        + "、".join(missing_cut_brands)
    )

shown_brands = display_df[BRAND_COL].dropna().astype(str).unique().tolist()
single_brand_lines = len(shown_brands) == 1
single_brand_cut = None
if single_brand_lines:
    b = shown_brands[0]
    row = brand_cut_df[brand_cut_df[BRAND_COL].astype(str) == b]
    if not row.empty:
        single_brand_cut = row.iloc[0]

if single_brand_cut is not None and pd.notna(single_brand_cut["x_cut"]) and pd.notna(single_brand_cut["y_cut"]):
    bx1, bx2, bx3 = st.columns(3)
    bx1.metric("目前品牌", str(single_brand_cut[BRAND_COL]))
    bx2.metric("品牌內 X 分界", f"{float(single_brand_cut['x_cut']):.2%}")
    bx3.metric("品牌內 Y 分界", f"{float(single_brand_cut['y_cut']):,.2f}")
else:
    st.info("目前畫面包含多個品牌，因此不顯示單一共用的十字分界線；請改看下方『品牌分界摘要』。")

hover_dict = {
    STORE_UI_COL: True,
    BRAND_COL: True,
    ZONE_COL: True,
    CITY_COL: True,
    CIRCLE_COL: True,
    "_x": ":.2%",
    "_y": ":,.2f",
    BRAND_X_CUT_COL: ":.2%",
    BRAND_Y_CUT_COL: ":,.2f",
    CALC_INCLUDED_COL: True,
    CALC_SAMPLE_N_COL: True,
    "象限": True,
}
if ADDRESS_COL in display_df.columns:
    hover_dict[ADDRESS_COL] = True
if DISTRICT_COL in display_df.columns:
    hover_dict[DISTRICT_COL] = True

plot_df = display_df.copy()
plot_df["_label"] = plot_df[STORE_UI_COL].fillna("").astype(str)
if show_labels and declutter_labels:
    x_rng = float(plot_df["_x"].max() - plot_df["_x"].min())
    y_rng = float(plot_df["_y"].max() - plot_df["_y"].min())
    x_bin = max(x_rng * (x_bin_pct / 100.0), 1e-9)
    y_bin = max(y_rng * (y_bin_pct / 100.0), 1e-9)
    tmp = plot_df[[STORE_UI_COL, "_x", "_y"]].copy()
    tmp["_xb"] = np.floor((tmp["_x"] - float(plot_df["_x"].min())) / x_bin).astype(int)
    tmp["_yb"] = np.floor((tmp["_y"] - float(plot_df["_y"].min())) / y_bin).astype(int)
    keep_idx = (
        tmp.assign(_idx=tmp.index)
        .sort_values(["_xb", "_yb", "_y"], ascending=[True, True, False])
        .groupby(["_xb", "_yb"], as_index=False)
        .head(1)["_idx"]
        .tolist()
    )
    plot_df["_label"] = np.where(plot_df.index.isin(keep_idx), plot_df["_label"], "")

fig = px.scatter(
    plot_df,
    x="_x",
    y="_y",
    color=BRAND_COL,
    text="_label",
    hover_data=hover_dict,
    labels={
        "_x": X_COL,
        "_y": Y_COL,
        STORE_UI_COL: "門店",
        BRAND_X_CUT_COL: "品牌內 X 分界",
        BRAND_Y_CUT_COL: "品牌內 Y 分界",
    },
    title=f"散點圖（品牌內{cut_mode}分界已預先計算｜顏色=品牌）",
)

if single_brand_cut is not None and pd.notna(single_brand_cut["x_cut"]) and pd.notna(single_brand_cut["y_cut"]):
    fig.add_vline(
        x=float(single_brand_cut["x_cut"]),
        line_dash="dash",
        annotation_text=f"X分界: {float(single_brand_cut['x_cut']):.2%}",
        annotation_position="top",
    )
    fig.add_hline(
        y=float(single_brand_cut["y_cut"]),
        line_dash="dash",
        annotation_text=f"Y分界: {float(single_brand_cut['y_cut']):,.2f}",
        annotation_position="right",
    )

if show_labels:
    fig.update_traces(
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=label_font_size),
        marker=dict(size=marker_size),
    )
else:
    fig.update_traces(
        mode="markers",
        textfont=dict(size=label_font_size),
        marker=dict(size=marker_size),
    )

fig.update_xaxes(tickformat=".0%")
fig.update_layout(template="plotly_dark", hovermode="closest", legend_title_text="品牌", height=900)
st.plotly_chart(fig, use_container_width=True)


# =========================
# Brand cutoff summary
# =========================
st.subheader("品牌分界摘要")
brand_display_cnt = (
    display_df.groupby(BRAND_COL, dropna=False)
    .size()
    .rename("目前顯示筆數")
    .reset_index()
)
brand_summary = (
    brand_cut_df.merge(brand_display_cnt, on=BRAND_COL, how="outer")
    .fillna({"calc_sample_n": 0, "目前顯示筆數": 0})
    .copy()
)
brand_summary["calc_sample_n"] = brand_summary["calc_sample_n"].astype(int)
brand_summary["目前顯示筆數"] = brand_summary["目前顯示筆數"].astype(int)

if brand_pick:
    brand_summary = brand_summary[brand_summary[BRAND_COL].astype(str).isin(brand_pick)].copy()

brand_summary = brand_summary.sort_values(["目前顯示筆數", BRAND_COL], ascending=[False, True])
brand_summary_view = brand_summary.copy()
brand_summary_view["品牌內 X 分界"] = brand_summary_view["x_cut"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
brand_summary_view["品牌內 Y 分界"] = brand_summary_view["y_cut"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
brand_summary_view = brand_summary_view.rename(columns={"calc_sample_n": "分界計算樣本數"})
brand_summary_view = brand_summary_view[[BRAND_COL, "分界計算樣本數", "目前顯示筆數", "品牌內 X 分界", "品牌內 Y 分界"]]
st.dataframe(brand_summary_view, width="stretch")


# =========================
# Quadrant counts
# =========================
st.subheader("各象限筆數（依目前顯示資料）")
q_meta = pd.DataFrame(
    [
        ["第一象限", "高營收, 高成長", "明星門市"],
        ["第二象限", "高營收, 低成長", "金牛門市"],
        ["第三象限", "低營收, 低成長", "問題門市"],
        ["第四象限", "低營收, 高成長", "潛力門市"],
        ["未分類", "無法計算品牌內分界", "待確認"],
    ],
    columns=["象限", "特徵", "店類型"],
)
q_cnt = display_df["象限"].value_counts().rename_axis("象限").reset_index(name="count")
q_table = q_meta.merge(q_cnt, on="象限", how="left").fillna({"count": 0})
q_table["count"] = q_table["count"].astype(int)
st.dataframe(q_table[["象限", "特徵", "店類型", "count"]], width="stretch")


# =========================
# Detail table
# =========================
st.subheader("門店明細（依目前顯示資料）")
detail_df = display_df.copy()
detail_df[CALC_INCLUDED_COL] = detail_df[CALC_INCLUDED_COL].map({True: "是", False: "否"})
detail_df[BRAND_X_CUT_COL] = detail_df[BRAND_X_CUT_COL].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
detail_df[BRAND_Y_CUT_COL] = detail_df[BRAND_Y_CUT_COL].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
detail_df[X_COL] = detail_df["_x"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
detail_df[Y_COL] = detail_df["_y"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")

detail_cols = [
    BRAND_COL,
    STORE_UI_COL,
    ZONE_COL,
    CITY_COL,
    DISTRICT_COL,
    ADDRESS_COL,
    CIRCLE_COL,
    X_COL,
    Y_COL,
    BRAND_X_CUT_COL,
    BRAND_Y_CUT_COL,
    CALC_INCLUDED_COL,
    CALC_SAMPLE_N_COL,
    "象限",
]
detail_cols_exist = [c for c in detail_cols if c in detail_df.columns]
st.dataframe(detail_df[detail_cols_exist].copy(), width="stretch")
