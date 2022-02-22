import streamlit as st

from streamlit_folium import folium_static

import datetime

import requests

import folium

import random

'''
# Projectwind model

This model predicts the energy production of 25 wind turbines
'''

WTG_longitude = st.number_input('pickup longitude', value=18.97466035850479)
WTG_latitude = st.number_input('pickup latitude', value=-69.27671861610212)

#prediction function

def pinpoint_WTGs(prediction = '0KW'):
    icon_url = "/Users/hamzabenkirane/Desktop/40963717-풍력-터빈-아이콘.webp"
    coordinates = [WTG_longitude,WTG_latitude]
    m = folium.Map(location=coordinates,zoom_start=12)
    for i in range(25):
        lat = random.uniform(18.96466035850479, 18.98466035850479)
        long = random.uniform(-69.26671861610212,-69.28671861610212)
        coordinates = [lat,long]
        folium.Marker(coordinates,popup=prediction,tooltip=f'WTG{i}',icon=folium.features.CustomIcon(icon_url,icon_size=(20,20))).add_to(m)
    return m

m = pinpoint_WTGs()
folium_static(m)
