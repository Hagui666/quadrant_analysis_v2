import io
import html
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

# 本/競品欄位候選
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

def find_first_existing_col(df_: pd.DataFrame, candidates):
    for c in candidates:
        if c in df_.columns:
            return c
    return None

def normalize_side(v: str) -> str:
    s = str(v).strip()
    if "本" in s:
        return "本品"
    if "競" in s:
        return "競品"
    return s

# =========================
# Validate required columns
# =========================
need_cols = [X_COL, Y_COL, BRAND_COL, INCLUDE_COL, MATCHED_COL]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.error(f"缺少欄位：{missing}\n目前欄位：{list(df.columns)}")
    st.stop()

# =========================
# Prepare base data (全資料，不做分區/城市等細分)
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

# 先找本/競品欄位（用全資料找，避免篩選後消失）
comp_col = find_first_existing_col(df, COMP_COL_CANDIDATES)

# =========================
# UI
# =========================
st.title("P2｜品牌象限分組（不做分區/城市等細分）")

cut_mode = st.radio("分界點計算方式", ["平均值", "中位數"], index=0, horizontal=True)

# ✅ 固定分界：永遠用「全資料 df」算，不受品牌選擇影響
df_all = df.copy()

if cut_mode == "平均值":
    x_cut = float(df_all["_x"].mean())
    y_cut = float(df_all["_y"].mean())
else:
    x_cut = float(df_all["_x"].median())
    y_cut = float(df_all["_y"].median())

conds = [
    (df_all["_x"] >= x_cut) & (df_all["_y"] >= y_cut),
    (df_all["_x"] <  x_cut) & (df_all["_y"] >= y_cut),
    (df_all["_x"] <  x_cut) & (df_all["_y"] <  y_cut),
    (df_all["_x"] >= x_cut) & (df_all["_y"] <  y_cut),
]
q_labels = ["第一象限", "第二象限", "第三象限", "第四象限"]
df_all["象限"] = np.select(conds, q_labels, default="未分類")

q_order_map = {"第一象限": 1, "第二象限": 2, "第三象限": 3, "第四象限": 4}
df_all["_q_order"] = df_all["象限"].map(q_order_map).fillna(99).astype(int)

# ✅ 品牌篩選：僅影響顯示（不影響象限計算）
brands = sorted(df_all[BRAND_COL].dropna().astype(str).unique().tolist())
brand_pick = st.multiselect(
    "品牌（多選，僅影響顯示，不影響象限分界/分類）",
    options=brands,
    default=brands
)

fdf = df_all[df_all[BRAND_COL].astype(str).isin(brand_pick)].copy() if brand_pick else df_all.iloc[0:0].copy()
if len(fdf) == 0:
    st.warning("目前品牌篩選結果為空，請調整選擇。")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("X 分界值（成長率）", f"{x_cut:.2%}")
c2.metric("Y 分界值（回推平均營業額）", f"{y_cut:,.2f}")
c3.metric("顯示筆數（篩選後）", f"{len(fdf):,}")

# =========================
# Output table (欄位跟 p1 明細一致 + 本/競品放第一欄)
# =========================
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
    X_COL,     # 成長率（X）
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

# =========================
# Quadrant dashboard (白底黑字)
# =========================
st.markdown("---")
st.subheader("象限儀表板（本品 / 競品）")

if comp_col is None:
    st.warning("找不到『本/競品』欄位，無法產生象限儀表板。請確認資料來源是否有本品/競品標記欄位（例如：本/競品）。")
else:
    q_meta = {
        "第一象限": {"desc": "高營收/高成長率", "tag": "明星"},
        "第二象限": {"desc": "高營收/低成長率", "tag": "金牛"},
        "第三象限": {"desc": "低營收/低成長率", "tag": "問題(觀察)"},
        "第四象限": {"desc": "低營收/高成長率", "tag": "潛力"},
    }

    # 位置：左上Q2、右上Q1、左下Q3、右下Q4
    grid_order = [
        "第二象限",
        "第一象限",
        "第三象限",
        "第四象限",
    ]

    def build_items(df_part: pd.DataFrame) -> str:
        if len(df_part) == 0:
            return '<div class="empty">（無）</div>'
        items = (df_part[BRAND_COL].astype(str).fillna("") + " " + df_part[STORE_UI_COL].astype(str).fillna("")).tolist()
        items = [html.escape(x.strip()) for x in items if x.strip()]
        return "<br>".join(items) if items else '<div class="empty">（無）</div>'

    dash_df = fdf.copy()
    dash_df["_side_norm"] = dash_df[comp_col].apply(normalize_side)

    css_white = """
    <style>
      body { background:#ffffff; color:#111111; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",Arial,sans-serif; }

      .quad-grid{
        display:grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
      }
      .quad{
        border: 1px solid rgba(0,0,0,0.18);
        border-radius: 10px;
        padding: 10px 12px;
        background: #ffffff;
      }
      .quad-title{
        text-align:center;
        font-weight: 800;
        font-size: 16px;
        margin-bottom: 4px;
        color:#111111;
      }
      .quad-sub{
        text-align:center;
        font-size: 12px;
        color:#333333;
        margin-bottom: 10px;
      }
      .inner-grid{
        display:grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .side{
        border: 1px solid rgba(0,0,0,0.16);
        border-radius: 8px;
        padding: 8px 10px;
        min-height: 140px;
        background: #ffffff;
      }
      .side-head{
        font-weight: 800;
        margin-bottom: 6px;
        color:#111111;
      }
      .tag-row{
        display:flex;
        align-items:center;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 6px;
        color:#111111;
      }
      .tag{ font-weight:800; }
      .count{ font-weight:800; color:#111111; }
      .items{
        font-size: 12px;
        line-height: 1.35;
        color:#111111;
        max-height: 220px;
        overflow: auto;
        padding-right: 4px;
      }
      .empty{ color:#666666; font-style: italic; }

      .items::-webkit-scrollbar { width: 8px; }
      .items::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.25); border-radius: 10px; }
      .items::-webkit-scrollbar-track { background: rgba(0,0,0,0.06); border-radius: 10px; }
    </style>
    """

    quad_parts = ['<div class="quad-grid">']

    for q_name in grid_order:
        desc = q_meta[q_name]["desc"]
        tag_name = q_meta[q_name]["tag"]

        df_q = dash_df[dash_df["象限"] == q_name].copy()
        df_q_ben = df_q[df_q["_side_norm"] == "本品"].copy()
        df_q_comp = df_q[df_q["_side_norm"] == "競品"].copy()

        ben_cnt = int(len(df_q_ben))
        comp_cnt = int(len(df_q_comp))

        ben_items = build_items(df_q_ben)
        comp_items = build_items(df_q_comp)

        quad_parts.append(
            f"""
            <div class="quad">
              <div class="quad-title">{q_name}</div>
              <div class="quad-sub">({desc})</div>

              <div class="inner-grid">
                <div class="side">
                  <div class="side-head">本品</div>
                  <div class="tag-row">
                    <div class="tag">{tag_name}：</div>
                    <div class="count">{ben_cnt}</div>
                  </div>
                  <div class="items">{ben_items}</div>
                </div>

                <div class="side">
                  <div class="side-head">競品</div>
                  <div class="tag-row">
                    <div class="tag">{tag_name}：</div>
                    <div class="count">{comp_cnt}</div>
                  </div>
                  <div class="items">{comp_items}</div>
                </div>
              </div>
            </div>
            """
        )

    quad_parts.append("</div>")

    html_block = css_white + "\n" + "\n".join(quad_parts)
    components.html(html_block, height=720, scrolling=True)