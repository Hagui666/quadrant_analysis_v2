import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="第三頁", layout="wide")

@st.cache_data(show_spinner=False)
def load_df_from_session(data_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data_bytes))

if "data_bytes" not in st.session_state:
    st.warning("尚未在入口頁上傳資料，請先回到 Home 上傳 Excel。")
    st.stop()

df = load_df_from_session(st.session_state["data_bytes"])

st.title("第三頁（規劃中）")
st.caption(f"目前資料筆數：{len(df):,}")

# TODO：你未來在這頁做自己的 filters / charts / tables