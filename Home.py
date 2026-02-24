import pandas as pd
import streamlit as st

st.set_page_config(page_title="分析入口", layout="wide")

# -------------------------
# Password gate (只在入口做)
# -------------------------
def require_password():
    try:
        app_pw = st.secrets["APP_PASSWORD"]
    except Exception:
        st.info("⚠ 目前為本機開發模式（未啟用密碼保護）")
        return True

    pw = st.text_input("請輸入存取密碼", type="password")
    if pw != app_pw:
        st.stop()
    return True

require_password()

# -------------------------
# Upload (只在入口做)
# -------------------------
st.title("商圈/門店分析入口")

uploaded = st.file_uploader("請上傳 Excel（.xlsx）作為本次分析資料來源", type=["xlsx"])

if uploaded is not None:
    st.session_state["data_bytes"] = uploaded.getvalue()
    st.success(f"已載入檔案：{uploaded.name}（本次瀏覽期間，所有頁面共用這份資料）")

# 若尚未上傳，提示使用者
if "data_bytes" not in st.session_state:
    st.info("請先上傳 Excel 後，再切換到左側 Pages 的各分析頁。")
    st.stop()

# -------------------------
# Optional: Quick preview / basic info
# -------------------------
@st.cache_data(show_spinner=False)
def read_df_from_bytes(b: bytes) -> pd.DataFrame:
    import io
    return pd.read_excel(io.BytesIO(b))

df = read_df_from_bytes(st.session_state["data_bytes"])
st.caption(f"資料筆數：{len(df):,}｜欄位數：{df.shape[1]}")

with st.expander("查看欄位清單"):
    st.write(list(df.columns))

st.markdown("---")
st.subheader("開始分析")
st.write("請從左側 Pages 選單進入：象限分析 / 第二頁 / 第三頁。")