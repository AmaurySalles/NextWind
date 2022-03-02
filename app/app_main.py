#app.py
import app
import app_turbine
import streamlit as st
st.set_page_config(
    page_title="ProjectWind", # => Quick reference - Streamlit
    page_icon=":serpent:",
    layout="wide", # wide
    initial_sidebar_state="collapsed") # collapsed
CSS = """
iframe {
    width: 100%;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
PAGES = {
    "Projectwind model": app,
    "Turbine details": app_turbine
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
