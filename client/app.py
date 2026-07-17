import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatUI import render_chat


st.set_page_config(page_title="AI medical Assistant",layout="wide")
st.title("🩺 medical Assistance chatbot")

render_chat()
render_uploader()
render_history_download()




