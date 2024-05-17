import streamlit as st

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {{
background: url("https://imgur.com/a/4BER9o3") no-repeat center center fixed;
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

