import streamlit as st
from utils.api import upload_pdf_api

def render_uploader():
    st.sidebar.header("upload Medical document(.PDFs)")
    uploaded_files=st.sidebar.file_uploader("upload Multiple PDF",type="pdf",accept_multiple_files=True)
    if st.sidebar.button("upload DB") and uploaded_files:
        response=upload_pdf_api(uploaded_files)
        if response.status_code==200:
            st.sidebar.success("uploaded successfully")
        else:
            st.sidebar.error(f"Error:{response.text}")    