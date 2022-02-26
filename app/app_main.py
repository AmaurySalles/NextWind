#app.py
import app
import app_turbine
import streamlit as st
PAGES = {
    "Projectwind model": app,
    "Turbine details": app_turbine
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
