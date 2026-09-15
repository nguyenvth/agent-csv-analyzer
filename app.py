import streamlit as st
import pandas as pd

st.set_page_config(page_title="Agent phân tích dữ liệu tự động", layout="wide")
st.title("Agent phân tích dữ liệu tự động")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Không đọc được file CSV: {e}")
        st.stop()

    st.success(f"Đã đọc file: {df.shape[0]} dòng, {df.shape[1]} cột")
    st.dataframe(df.head(20))

    st.subheader("Nhập mục tiêu phân tích")
    goal = st.text_area("Mô tả bằng tiếng Việt bạn muốn phân tích gì từ dữ liệu này")

    if st.button("Xác nhận"):
        st.write("Mục tiêu đã nhận:", goal)
        st.info("Bước tiếp theo (agent xử lý) sẽ được thêm sau.")
else:
    st.info("Vui lòng upload file CSV để bắt đầu.")