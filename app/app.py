import streamlit as st

from streamlit_folium import folium_static

import datetime

import requests

import folium
import time
import random
#WTG_longitude = st.number_input('pickup longitude', value=18.97466035850479)
#WTG_latitude = st.number_input('pickup latitude', value=-69.27671861610212)

def pinpoint_WTGs(prediction = '0KW'):
    WTG_longitude = 18.97466035850479
    WTG_latitude= -69.27671861610212
    icon_url = "40963717-vento-icona-turbine.webp"
    coordinates = [WTG_longitude,WTG_latitude]
    m = folium.Map(location=coordinates,zoom_start=14.5, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community')
    for i in range(25):
        lat = random.uniform(18.96466035850479, 18.98466035850479)
        long = random.uniform(-69.26671861610212,-69.28671861610212)
        coordinates = [lat,long]
        folium.Marker(coordinates,popup=prediction,tooltip=f'WTG{i}',icon=folium.features.CustomIcon(icon_url,icon_size=(20,20))).add_to(m)
    return m

def pinpoint_WTGs_prediction(prediction = '0KW'):
    WTG_longitude = 18.97466035850479
    WTG_latitude= -69.27671861610212
    icon_url = "40963717-vento-icona-turbine.webp"
    coordinates = [WTG_longitude,WTG_latitude]
    m = folium.Map(location=coordinates,zoom_start=14.5, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community')
    for i in range(25):
        lat = random.uniform(18.96466035850479, 18.98466035850479)
        long = random.uniform(-69.26671861610212,-69.28671861610212)
        coordinates = [lat,long]
        folium.Marker(coordinates,popup=prediction,tooltip=f'WTG{i}',icon=folium.features.CustomIcon(icon_url,icon_size=(20,20))).add_to(m)
    return m
def app():
    st.write('''
    # Projectwind model

    This model predicts the energy production of 25 wind turbines
    ''')

    option = st.selectbox(
        'From which wind power station would you like to predict the energy produced?',
        ('None', 'Station 1', 'Station 2'))

    if option == 'Station 1':
        button = st.button('Predict')
        if button:
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.write('Power generated for the next 12h: XXXXXXX')
        else:
            st.write('')

        #prediction function
        m = pinpoint_WTGs()
        folium_static(m)

    elif option=='Station 2':
        if st.button('Predict'):
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.write('Power generated for the next 12h: XXXXXXX')
        else:
            st.write('')

        m = pinpoint_WTGs()

        folium_static(m)
