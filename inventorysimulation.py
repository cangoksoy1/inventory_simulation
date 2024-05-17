import streamlit as st

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background: url("https://i.imgur.com/kox6xPx.png");
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
