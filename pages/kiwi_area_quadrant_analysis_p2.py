import io
import html
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import json
import math

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
# Prepare base data (全資料；不做分區/城市篩選)
# =========================
df = df.copy()

df[INCLUDE_COL] = df[INCLUDE_COL].astype(str).str.strip().str.lower()
df = df[df[INCLUDE_COL] == "y"].copy()

df[STORE_UI_COL] = (
    df[MATCHED_COL].astype(str)
    .replace({"nan": np.nan})
    .fillna("")
    .str.strip()
    .replace({"": np.nan})
)

df["_x_raw"] = to_numeric_series(df[X_COL])  # 成長率
df["_y_raw"] = to_numeric_series(df[Y_COL])  # 回推平均營業額
df = df.dropna(subset=["_x_raw", "_y_raw"]).copy()

# 成長率逐筆偵測：32.15 -> 0.3215（與 p1 一致）
df["_x"] = df["_x_raw"].where(df["_x_raw"].abs() <= 1.5, df["_x_raw"] / 100.0)
df["_y"] = df["_y_raw"]

comp_col = find_first_existing_col(df, COMP_COL_CANDIDATES)

# =========================
# UI
# =========================
st.title("P2｜品牌象限分組（每個品牌各自計算分界）")

cut_mode = st.radio("分界點計算方式", ["平均值", "中位數"], index=0, horizontal=True)

brands_all = sorted(df[BRAND_COL].dropna().astype(str).unique().tolist())
brand_pick = st.multiselect(
    "品牌（多選，僅影響顯示，不影響各品牌分組運算）",
    options=brands_all,
    default=brands_all
)

# =========================
# Brand color map (全頁一致：同品牌在不同區塊顏色固定)
# =========================
PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
brand_color_map_global = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(brands_all)}

# =========================
# 1) 先用「全資料 df」算出每個品牌自己的分界值（固定，不受 brand_pick 影響）
# =========================
if cut_mode == "平均值":
    brand_cuts = (
        df.groupby(BRAND_COL, dropna=False)
          .agg(brand_x_cut=("_x", "mean"), brand_y_cut=("_y", "mean"), store_cnt=("_x", "size"))
          .reset_index()
    )
else:
    brand_cuts = (
        df.groupby(BRAND_COL, dropna=False)
          .agg(brand_x_cut=("_x", "median"), brand_y_cut=("_y", "median"), store_cnt=("_x", "size"))
          .reset_index()
    )

# =========================
# 2) 把品牌分界 merge 回每筆資料，計算「品牌內象限」
# =========================
df_all = df.merge(brand_cuts[[BRAND_COL, "brand_x_cut", "brand_y_cut"]], on=BRAND_COL, how="left")

conds = [
    (df_all["_x"] >= df_all["brand_x_cut"]) & (df_all["_y"] >= df_all["brand_y_cut"]),
    (df_all["_x"] <  df_all["brand_x_cut"]) & (df_all["_y"] >= df_all["brand_y_cut"]),
    (df_all["_x"] <  df_all["brand_x_cut"]) & (df_all["_y"] <  df_all["brand_y_cut"]),
    (df_all["_x"] >= df_all["brand_x_cut"]) & (df_all["_y"] <  df_all["brand_y_cut"]),
]
q_labels = ["第一象限", "第二象限", "第三象限", "第四象限"]
df_all["象限"] = np.select(conds, q_labels, default="未分類")

q_order_map = {"第一象限": 1, "第二象限": 2, "第三象限": 3, "第四象限": 4}
df_all["_q_order"] = df_all["象限"].map(q_order_map).fillna(99).astype(int)

# =========================
# 各品牌分界值表（只顯示勾選品牌；並新增四象限筆數欄）
# =========================
brand_cuts_show = brand_cuts[brand_cuts[BRAND_COL].astype(str).isin(brand_pick)].copy() if brand_pick else brand_cuts.iloc[0:0].copy()
brand_cuts_show = brand_cuts_show.sort_values(BRAND_COL)

# 每品牌四象限筆數（以品牌內象限分類後 df_all 統計）
q_cols_order = ["第一象限", "第二象限", "第三象限", "第四象限"]
brand_q_counts = (
    df_all.groupby([BRAND_COL, "象限"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=q_cols_order, fill_value=0)
          .reset_index()
)

st.subheader("各品牌分界值（用於該品牌門店象限判定）")
show_cuts = brand_cuts_show.copy()
show_cuts["X分界(成長率)"] = show_cuts["brand_x_cut"].map(lambda v: f"{v:.2%}")
show_cuts["Y分界(回推平均營業額)"] = show_cuts["brand_y_cut"].map(lambda v: f"{v:,.2f}")
show_cuts = show_cuts[[BRAND_COL, "store_cnt", "X分界(成長率)", "Y分界(回推平均營業額)"]].rename(columns={"store_cnt": "筆數"})

# merge 四象限計數
show_cuts = show_cuts.merge(brand_q_counts, on=BRAND_COL, how="left").fillna(0)
for c in q_cols_order:
    show_cuts[c] = show_cuts[c].astype(int)

# 欄位順序：品牌、筆數、X分界、Y分界、Q1~Q4
show_cuts = show_cuts[[BRAND_COL, "筆數", "X分界(成長率)", "Y分界(回推平均營業額)"] + q_cols_order]
st.dataframe(show_cuts, width="stretch")

# ✅ 最後才套 brand_pick 做「顯示過濾」
fdf = df_all[df_all[BRAND_COL].astype(str).isin(brand_pick)].copy() if brand_pick else df_all.iloc[0:0].copy()
if len(fdf) == 0:
    st.warning("目前品牌篩選結果為空，請調整選擇。")
    st.stop()

st.caption(f"顯示筆數（篩選後）：{len(fdf):,}")

# =========================
# Output table (欄位跟 p1 明細一致 + 本/競品放第一欄)
# =========================
detail_cols = []
if comp_col is not None:
    detail_cols.append(comp_col)

detail_cols += [
    BRAND_COL,
    STORE_UI_COL,
    ZONE_COL,
    CITY_COL,
    "行政區",
    ADDRESS_COL,
    CIRCLE_COL,
    Y_COL,
    X_COL,
    "象限",
]

detail_cols_exist = [c for c in detail_cols if c in fdf.columns]

out_df = (
    fdf.sort_values([BRAND_COL, "_q_order"])
       .loc[:, detail_cols_exist]
       .copy()
)

st.subheader("品牌內象限分組後門店明細（含象限）")
st.dataframe(out_df, width="stretch")

# =========================
# Quadrant dashboard (白底黑字 + 不使用內部捲軸 + 可下載 PNG)
# =========================
st.markdown("---")
st.subheader("象限儀表板（本品 / 競品）")
height_multiplier = st.slider("儀表板高度倍率（避免截斷）", min_value=0.2, max_value=3.0, value=1.6, step=0.05)

if comp_col is None:
    st.warning("找不到『本/競品』欄位，無法產生象限儀表板。")
else:
    q_meta = {
        "第一象限": {"desc": "高營收/高成長率", "tag": "明星"},
        "第二象限": {"desc": "高營收/低成長率", "tag": "金牛"},
        "第三象限": {"desc": "低營收/低成長率", "tag": "問題(觀察)"},
        "第四象限": {"desc": "低營收/高成長率", "tag": "潛力"},
    }

    # 位置：左上Q2、右上Q1、左下Q3、右下Q4
    grid_order = ["第二象限", "第一象限", "第三象限", "第四象限"]

    def build_items(df_part: pd.DataFrame, brand_color_map: dict) -> str:
        """
        產生「只顯示門店名」的清單，並用品牌顏色的圓點做前綴，方便視覺辨識。
        """
        if len(df_part) == 0:
            return '<div class="empty">（無）</div>'

        stores = df_part[STORE_UI_COL].astype(str).fillna("").tolist()
        brands = df_part[BRAND_COL].astype(str).fillna("").tolist()

        lis_parts = []
        for b, s in zip(brands, stores):
            s = (s or "").strip()
            if not s:
                continue
            color = brand_color_map.get(b, "#111111")
            lis_parts.append(
                f'<li title="{html.escape(str(b))}">'
                f'<span class="dot" style="background:{html.escape(color)}"></span>'
                f'<span class="name">{html.escape(s)}</span>'
                f'</li>'
            )

        if not lis_parts:
            return '<div class="empty">（無）</div>'

        return '<ul class="items">' + "".join(lis_parts) + '</ul>'

    dash_df = fdf.copy()
    dash_df["_side_norm"] = dash_df[comp_col].apply(normalize_side)

    # 依品牌建立固定顏色對應（全頁一致）
    uniq_brands = sorted(dash_df[BRAND_COL].dropna().astype(str).unique().tolist())
    brand_color_map = brand_color_map_global

    # 圖例（品牌 ↔ 顏色）
    legend_items = []
    for b in uniq_brands:
        c = brand_color_map.get(b, "#111111")
        legend_items.append(
            f'<div class="legend-item"><span class="dot" style="background:{html.escape(c)}"></span>{html.escape(str(b))}</div>'
        )
    legend_html = '<div class="legend">' + ''.join(legend_items) + '</div>' if legend_items else ''

    # 估算 iframe 高度（保守估計，避免內容被截斷）
    # 目前清單是兩欄顯示，所以以「行數 = ceil(筆數/2)」估算高度
    def _rows(n: int) -> int:
        return int((n + 1) // 2)  # ceil(n/2) for int

    line_px = 26      # 每行高度（含行距/圓點）
    quad_base = 190   # 每個象限的固定高度（標題/邊界/間距/圖例等）
    # 四象限各自取本品/競品較大筆數（兩欄顯示）
    quad_rows = {}
    for q in ["第一象限", "第二象限", "第三象限", "第四象限"]:
        ben_cnt = dash_df[(dash_df["象限"] == q) & (dash_df["_side_norm"] == "本品")].shape[0]
        comp_cnt = dash_df[(dash_df["象限"] == q) & (dash_df["_side_norm"] == "競品")].shape[0]
        quad_rows[q] = _rows(max(ben_cnt, comp_cnt))

    top_rows = max(quad_rows.get("第二象限", 0), quad_rows.get("第一象限", 0))
    bot_rows = max(quad_rows.get("第三象限", 0), quad_rows.get("第四象限", 0))

    estimated_height = 260 + (quad_base + top_rows * line_px) + (quad_base + bot_rows * line_px)
    estimated_height = max(1100, int(estimated_height))

    # 由 JS 自動回傳高度（避免白底區塊過長或被截斷），此處不再手動估算 iframe 高度。

    css_white = """
    <style>
      body { background:#ffffff; color:#111111; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",Arial,sans-serif; }

      .toolbar{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }
      .btn{ background:#111; color:#fff; border:none; border-radius:8px; padding:8px 10px; font-weight:700; cursor:pointer; }
      .btn:hover{ opacity:0.9; }
      .hint{ font-size:12px; color:#444; }

      .quad-grid{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .quad{ border:1px solid rgba(0,0,0,0.18); border-radius:10px; padding:10px 12px; background:#fff; }
      .quad-title{ text-align:center; font-weight:800; font-size:16px; margin-bottom:4px; color:#111; }
      .quad-sub{ text-align:center; font-size:12px; color:#333; margin-bottom:10px; }
      .inner-grid{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
      .side{ border:1px solid rgba(0,0,0,0.16); border-radius:8px; padding:8px 10px; min-height:90px; background:#fff; }
      .side-head{ font-weight:800; margin-bottom:6px; color:#111; }
      .tag-row{ display:flex; align-items:center; justify-content:space-between; font-size:13px; margin-bottom:6px; color:#111; }
      .tag{ font-weight:800; }
      .count{ font-weight:800; color:#111; }

      /* ✅ 兩欄條列：節省高度 + 字體放大（方便貼簡報） */
      #dashboard{height:auto; overflow:visible;}

      .legend{
        display:flex;
        flex-wrap: wrap;
        gap: 10px 14px;
        align-items:center;
        margin: 6px 0 12px 0;
        padding: 8px 10px;
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 10px;
        background: #ffffff;
      }
      .legend-item{
        display:flex;
        align-items:center;
        gap: 7px;
        font-size: 12px;
        color:#111;
        white-space: nowrap;
      }

      .items{
        list-style: none;     /* 取消預設黑點 */
        padding: 0;
        margin: 0;
        font-size: 13px;      /* 稍大、但不會太擠 */
        line-height: 1.55;
        color:#111;

        /* ✅ 多直列顯示（縮短高度） */
        column-count: 2;
        column-gap: 18px;
      }
      .items li{
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0 0 6px 0;
        break-inside: avoid;
        -webkit-column-break-inside: avoid;

        /* ✅ 每筆一行不換行 */
        white-space: nowrap;
      }
      .dot{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
        flex: 0 0 9px;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.15);
      }
      .name{
        display: inline-block;
      }

      /* 若門店名稱很長：不換行，改用左右捲動（避免截斷或換行） */
      .side{
        overflow-x: auto;
      }

      .empty{ color:#666; font-style:italic; }
    </style>
    """

    quad_parts = [
        f"""
        <div class="toolbar">
          <button class="btn" onclick="downloadDashboard()">下載圖片（PNG）</button>
          <span class="hint">（會將下方整個象限儀表板輸出成圖片）</span>
        </div>
        <div id="dashboard">
          {legend_html}
          <div class="quad-grid">
        """
    ]

    for q_name in grid_order:
        desc = q_meta[q_name]["desc"]
        tag_name = q_meta[q_name]["tag"]

        df_q = dash_df[dash_df["象限"] == q_name].copy()
        df_q_ben = df_q[df_q["_side_norm"] == "本品"].copy()
        df_q_comp = df_q[df_q["_side_norm"] == "競品"].copy()

        ben_cnt = int(len(df_q_ben))
        comp_cnt = int(len(df_q_comp))

        ben_items = build_items(df_q_ben, brand_color_map)
        comp_items = build_items(df_q_comp, brand_color_map)

        quad_parts.append(
            f"""
            <div class="quad">
              <div class="quad-title">{q_name}</div>
              <div class="quad-sub">({desc})</div>
              <div class="inner-grid">
                <div class="side">
                  <div class="side-head">本品</div>
                  <div class="tag-row"><div class="tag">{tag_name}：</div><div class="count">{ben_cnt}</div></div>
                  <div class="items">{ben_items}</div>
                </div>
                <div class="side">
                  <div class="side-head">競品</div>
                  <div class="tag-row"><div class="tag">{tag_name}：</div><div class="count">{comp_cnt}</div></div>
                  <div class="items">{comp_items}</div>
                </div>
              </div>
            </div>
            """
        )

    quad_parts.append("</div></div>")  # close quad-grid + dashboard

    script = """
    <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
    <script>
      // === Streamlit iframe auto-resize ===
      function reportHeight(){
        try{
          const h = Math.max(
            document.documentElement.scrollHeight,
            document.body.scrollHeight,
            document.documentElement.offsetHeight,
            document.body.offsetHeight
          );
          // Streamlit 內建訊息格式（components.html 也適用）
          window.parent.postMessage(
            {isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: h},
            "*"
          );
        }catch(e){}
      }

      // 初次與資源載入後回報
      window.addEventListener("load", () => { reportHeight(); setTimeout(reportHeight, 200); setTimeout(reportHeight, 800); });

      // 內容變動即回報（字體/清單變動/展開收合）
      const ro = new ResizeObserver(() => { reportHeight(); });
      ro.observe(document.body);

      // === PNG download ===
      function downloadDashboard(){
        const el = document.getElementById('dashboard');
        if(!el){ alert('找不到儀表板'); return; }

        // 暫存原樣式
        const prevOverflow = el.style.overflow;
        const prevHeight = el.style.height;
        const prevWidth  = el.style.width;

        // 讓內容完整展開再截圖（避免只截到可視區）
        el.style.overflow = 'visible';
        el.style.height = 'auto';
        el.style.width  = 'auto';

        // 取得完整內容尺寸（含超出可視範圍）
        const fullH = Math.max(el.scrollHeight, el.offsetHeight);
        const fullW = Math.max(el.scrollWidth,  el.offsetWidth);

        html2canvas(el, {
          backgroundColor: '#ffffff',
          scale: 2,
          width: fullW,
          height: fullH,
          windowWidth: fullW,
          windowHeight: fullH,
          scrollX: 0,
          scrollY: 0
        }).then(canvas => {
          const link = document.createElement('a');
          link.download = '象限儀表板.png';
          link.href = canvas.toDataURL('image/png');
          link.click();

          // 還原樣式
          el.style.overflow = prevOverflow;
          el.style.height = prevHeight;
          el.style.width  = prevWidth;
        }).catch(err => {
          el.style.overflow = prevOverflow;
          el.style.height = prevHeight;
          el.style.width  = prevWidth;
          console.error(err);
          alert('截圖失敗，請稍後再試');
        });
      }
    </script>
    """

    html_block = css_white + "\n" + "\n".join(quad_parts) + "\n" + script

    # scrolling=False：避免 iframe 自己出現捲動條
    components.html(html_block, height=int(estimated_height * height_multiplier), scrolling=False)

# =========================
# Area/City/Circle table under dashboard (HTML table)
# =========================
st.markdown("---")
st.subheader("分區 × 城市 × 商圈 象限彙整表（本品 / 競品）")

render_mode = st.radio(
    "呈現方式",
    ["互動表格（可下載CSV）", "儀表板樣式（HTML，可下載PNG）"],
    index=0,
    horizontal=True,
    help="互動表格可直接在 Streamlit 顯示並下載 CSV；HTML 模式可模擬 Excel 合併儲存格並下載 PNG。"
)


need_area_cols = [ZONE_COL, CITY_COL, CIRCLE_COL]
miss_area = [c for c in need_area_cols if c not in fdf.columns]
if miss_area:
    st.warning(f"缺少欄位 {miss_area}，無法產生分區/城市/商圈彙整表。")
else:
    # 使用顯示後資料 fdf（已套 brand_pick），並沿用象限與本/競品正規化
    area_df = fdf.copy()
    if comp_col is None:
        st.warning("找不到『本/競品』欄位，無法產生分區/城市/商圈彙整表。")
    else:
        area_df["_side_norm"] = area_df[comp_col].apply(normalize_side)

        def _safe_str(v):
            if pd.isna(v):
                return ""
            return str(v).strip()

        def _join_stores(df_part: pd.DataFrame, brand_color_map: dict) -> str:
            """Join stores into HTML lines, with a colored dot indicating the store's brand."""
            if df_part is None or len(df_part) == 0:
                return ""

            stores = (
                df_part[[BRAND_COL, STORE_UI_COL]]
                .dropna(subset=[STORE_UI_COL])
                .assign(_store=lambda d: d[STORE_UI_COL].astype(str).map(lambda s: s.strip()))
                .replace({"_store": {"": np.nan}})
                .dropna(subset=["_store"])
            )

            # 去重但保序：以 (品牌, 門店) 為鍵
            seen = set()
            parts = []
            for b, s in zip(stores[BRAND_COL].astype(str).fillna(""), stores["_store"].tolist()):
                key = (b, s)
                if key in seen:
                    continue
                seen.add(key)
                color = brand_color_map.get(b, "#111111")
                parts.append(
                    f'<div class="store" title="{html.escape(str(b))}">'
                    f'<span class="dot" style="background:{html.escape(color)}"></span>'
                    f'<span class="name">{html.escape(s)}</span>'
                    f'</div>'
                )

            return "".join(parts)

        # 以「分區→城市→商圈」排序
        area_df[ZONE_COL] = area_df[ZONE_COL].apply(_safe_str)
        area_df[CITY_COL] = area_df[CITY_COL].apply(_safe_str)
        area_df[CIRCLE_COL] = area_df[CIRCLE_COL].apply(_safe_str)

        area_df = area_df.sort_values([ZONE_COL, CITY_COL, CIRCLE_COL, BRAND_COL, STORE_UI_COL]).copy()

        # 先建立每個（分區,城市,商圈）對應的內容
        group_keys = [ZONE_COL, CITY_COL, CIRCLE_COL]
        rows = []
        for (z, c, cir), g in area_df.groupby(group_keys, dropna=False):
            # 店數
            cnt_ben = int((g["_side_norm"] == "本品").sum())
            cnt_comp = int((g["_side_norm"] == "競品").sum())

            # 四象限文字清單（以門店欄位顯示）
            cell = {}
            for q in ["第一象限", "第二象限", "第三象限", "第四象限"]:
                for side in ["本品", "競品"]:
                    gg = g[(g["象限"] == q) & (g["_side_norm"] == side)]
                    cell[(q, side)] = _join_stores(gg, brand_color_map_global)

            rows.append({
                ZONE_COL: z,
                CITY_COL: c,
                CIRCLE_COL: cir,
                ("店數", "本品"): cnt_ben,
                ("店數", "競品"): cnt_comp,
                ("第一象限", "本品"): cell[("第一象限", "本品")],
                ("第一象限", "競品"): cell[("第一象限", "競品")],
                ("第二象限", "本品"): cell[("第二象限", "本品")],
                ("第二象限", "競品"): cell[("第二象限", "競品")],
                ("第三象限", "本品"): cell[("第三象限", "本品")],
                ("第三象限", "競品"): cell[("第三象限", "競品")],
                ("第四象限", "本品"): cell[("第四象限", "本品")],
                ("第四象限", "競品"): cell[("第四象限", "競品")],
            })

        if len(rows) == 0:
            st.info("目前資料無法產生分區/城市/商圈彙整表（可能篩選後為空）。")
        else:
            table_df = pd.DataFrame(rows)

            # 計算 rowspan（模擬 Excel 合併儲存格）
            # 以排序後 table_df 的連續區段計算
            table_df = table_df.sort_values([ZONE_COL, CITY_COL, CIRCLE_COL]).reset_index(drop=True)

            zone_spans = [0] * len(table_df)
            city_spans = [0] * len(table_df)

            # zone spans
            i = 0
            while i < len(table_df):
                z = table_df.loc[i, ZONE_COL]
                j = i
                while j < len(table_df) and table_df.loc[j, ZONE_COL] == z:
                    j += 1
                span = j - i
                zone_spans[i] = span
                for k in range(i + 1, j):
                    zone_spans[k] = 0
                i = j

            # city spans within zone
            i = 0
            while i < len(table_df):
                z = table_df.loc[i, ZONE_COL]
                c = table_df.loc[i, CITY_COL]
                j = i
                while j < len(table_df) and table_df.loc[j, ZONE_COL] == z and table_df.loc[j, CITY_COL] == c:
                    j += 1
                span = j - i
                city_spans[i] = span
                for k in range(i + 1, j):
                    city_spans[k] = 0
                i = j

            header_groups = [
                ("店數", ""),
                ("第一象限", "（高營收/高成長）"),
                ("第二象限", "（高營收/低成長）"),
                ("第三象限", "（低營收/低成長）"),
                ("第四象限", "（低營收/高成長）"),
            ]

            css_table = """
            <style>
              .area-wrap{
                background:#fff;
                color:#111;
                font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",Arial,sans-serif;

                /* ✅ 垂直捲動：表格較長時不會被截斷 */
                overflow-x: hidden;
                border: 1px solid rgba(0,0,0,0.15);
                border-radius: 10px;
              }
              .area-scroll{
                /* ✅ 只讓表格區滾動（不含工具列） */
                max-height: 70vh;   /* 可依需求調整：如 600px / 90vh */
                overflow-y: auto;
              }
              .area-toolbar{ display:flex; align-items:center; gap:10px; margin: 4px 0 10px 0; }
              .area-btn{ background:#111; color:#fff; border:none; border-radius:8px; padding:8px 10px; font-weight:700; cursor:pointer; }
              .area-btn:hover{ opacity:0.9; }
              .area-hint{ font-size:12px; color:#444; }

              table.area{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
                font-size: 11px;
              }
              table.area th, table.area td{
                border: 1px solid rgba(0,0,0,0.28);
                padding: 6px 6px;
                vertical-align: top;
                word-break: break-word;
                background: #fff;
              }
              table.area th{
                text-align: center;
                font-weight: 800;
              }
              .sticky-head thead th{
                position: sticky;
                top: 0;
                z-index: 2;
              }
              td.merge{
                text-align: center;
                vertical-align: middle;
                font-weight: 800;
                background: #fafafa;
              }
              td.circle{
                font-weight: 700;
                background: #ffffff;
              }
              td.num{
                text-align:center;
                vertical-align: middle;
                font-weight: 800;
                background:#ffffff;
              }
              td.cell{
                line-height: 1.55;
                white-space: normal;
              }

              .store{
                display:flex;
                align-items:flex-start;
                gap:6px;
                margin: 0 0 4px 0;
                break-inside: avoid;
              }
              .store .dot{
                width: 9px;
                height: 9px;
                border-radius: 50%;
                display:inline-block;
                flex: 0 0 9px;
                margin-top: 4px;
                box-shadow: 0 0 0 1px rgba(0,0,0,0.15);
              }
              .store .name{
                display:inline-block;
              }
              .muted{ color:#666; font-style: italic; }

              /* column widths (縮窄以避免水平捲動) */
              col.z{ width: 56px; }
              col.city{ width: 72px; }
              col.circle{ width: 140px; }
              col.small{ width: 50px; }
              col.big{ width: 120px; }
            </style>
            """

            # Header rows
            h1 = []
            h2 = []

            h1.append('<th rowspan="2">分區編碼</th>')
            h1.append('<th rowspan="2">城市</th>')
            h1.append('<th rowspan="2">商圈名稱(kiwi)</th>')

            for gname, gdesc in header_groups:
                title = html.escape(gname + (gdesc or ""))
                h1.append(f'<th colspan="2">{title}</th>')
                h2.append('<th>本品</th><th>競品</th>')

            thead = "<thead><tr>" + "".join(h1) + "</tr><tr>" + "".join(h2) + "</tr></thead>"

            colgroup = """
            <colgroup>
              <col class="z"/><col class="city"/><col class="circle"/>
              <col class="small"/><col class="small"/>
              <col class="big"/><col class="big"/>
              <col class="big"/><col class="big"/>
              <col class="big"/><col class="big"/>
              <col class="big"/><col class="big"/>
            </colgroup>
            """

            body_rows = []
            for idx, r in table_df.iterrows():
                tds = []
                if zone_spans[idx] > 0:
                    tds.append(f'<td class="merge" rowspan="{zone_spans[idx]}">{html.escape(str(r[ZONE_COL]))}</td>')
                if city_spans[idx] > 0:
                    tds.append(f'<td class="merge" rowspan="{city_spans[idx]}">{html.escape(str(r[CITY_COL]))}</td>')

                tds.append(f'<td class="circle">{html.escape(str(r[CIRCLE_COL]))}</td>')

                tds.append(f'<td class="num">{int(r[("店數","本品")])}</td>')
                tds.append(f'<td class="num">{int(r[("店數","競品")])}</td>')

                for q in ["第一象限","第二象限","第三象限","第四象限"]:
                    for side in ["本品","競品"]:
                        v = r[(q, side)]
                        if not isinstance(v, str) or v.strip() == "":
                            tds.append('<td class="cell"></td>')
                        else:
                            tds.append(f'<td class="cell">{v}</td>')

                body_rows.append("<tr>" + "".join(tds) + "</tr>")

            tbody = "<tbody>" + "".join(body_rows) + "</tbody>"

            script_table = """
            <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
            <script>
              function reportHeight(){
                try{
                  const h = Math.max(
                    document.documentElement.scrollHeight,
                    document.body.scrollHeight,
                    document.documentElement.offsetHeight,
                    document.body.offsetHeight
                  );
                  window.parent.postMessage(
                    {isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: h},
                    "*"
                  );
                }catch(e){}
              }
              window.addEventListener("load", () => { reportHeight(); setTimeout(reportHeight, 200); setTimeout(reportHeight, 800); });
              const ro = new ResizeObserver(() => { reportHeight(); });
              ro.observe(document.body);

              function downloadAreaTable(){
                const el = document.getElementById('areaTableWrap');
                if(!el){ alert('找不到表格'); return; }
                const fullH = Math.max(el.scrollHeight, el.offsetHeight);
                const fullW = Math.max(el.scrollWidth,  el.offsetWidth);
                html2canvas(el, {
                  backgroundColor: '#ffffff',
                  scale: 2,
                  width: fullW,
                  height: fullH,
                  windowWidth: fullW,
                  windowHeight: fullH,
                  scrollX: 0,
                  scrollY: 0
                }).then(canvas => {
                  const link = document.createElement('a');
                  link.download = '分區城市商圈象限表.png';
                  link.href = canvas.toDataURL('image/png');
                  link.click();
                }).catch(err => {
                  console.error(err);
                  alert('截圖失敗，請稍後再試');
                });
              }
            </script>
            """

            html_table = (
                css_table +
                '<div class="area-wrap" id="areaTableWrap">' +
                '<div class="area-toolbar">' +
                '<button class="area-btn" onclick="downloadAreaTable()">下載表格（PNG）</button>' +
                '<span class="area-hint">（會將下方整張表輸出成圖片）</span>' +
                '</div>' +
                '<div class="area-scroll">' +
                f'<table class="area sticky-head">{colgroup}{thead}{tbody}</table>' +
                '</div>' +
                '</div>' +
                script_table
            )

            # 互動表格（Streamlit DataFrame）或 HTML（合併儲存格+PNG）
            if render_mode.startswith("互動表格"):
                # 模擬 Excel 合併儲存格：同分區/同城市的重複值留空
                disp = table_df.copy()

                # 依連續重複區段把後續列清空（視覺上像合併）
                for col in [ZONE_COL, CITY_COL]:
                    last = None
                    for r in range(len(disp)):
                        v = disp.loc[r, col]
                        if last is None:
                            last = v
                        else:
                            if v == last:
                                disp.loc[r, col] = ""
                            else:
                                last = v

                st.dataframe(disp, use_container_width=True, hide_index=True)

                # CSV 下載（扁平化雙層欄位）
                dl = disp.copy()
                if isinstance(dl.columns, pd.MultiIndex):
                    dl.columns = [f"{a}-{b}" if b else str(a) for a, b in dl.columns]
                csv_bytes = dl.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "下載表格（CSV）",
                    data=csv_bytes,
                    file_name="area_city_circle_quadrant_table.csv",
                    mime="text/csv"
                )
            else:
                est_h = int(160 + len(table_df) * 34)
                est_h = max(520, min(est_h, 2200))
                components.html(html_table, height=est_h, scrolling=False)
