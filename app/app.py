import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import datetime
import datetime

import requests

import folium
import time
import random

from projectwind.predict import load_and_predict

#WTG_longitude = st.number_input('pickup longitude', value=18.97466035850479)
#WTG_latitude = st.number_input('pickup latitude', value=-69.27671861610212)

def randomise_turbine_loc():
    points = []
    for i in range(25):
        lat = random.uniform(18.95, 18.99)
        long = random.uniform(-69.25,-69.29)
        coordinates = [lat,long]
        points.append(coordinates)
    return points

def pinpoint_WTGs_prediction():
    
    # Load predictions
    predictions = load_and_predict()
    predictions_25 = predictions[0:25]
    
    # Load WTG locations
    locations = randomise_turbine_loc()

    # Site location
    WTG_longitude = 18.9746
    WTG_latitude= -69.2767

    # Generate map
    coordinates = [WTG_longitude,WTG_latitude]
    m = folium.Map(location=coordinates,zoom_start=13.5 ,tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community')
  
    # Generate turbine icons
    icon_url = "app/wind-turbine.png"
    icon_url_red = "app/wind-turbine-red.png"
    icon_url_orange = "app/wind-turbine-orange.png"
    for i in locations:
        if predictions_25[locations.index(i)][0]==2:
            folium.Marker(i,popup="High",tooltip=f'WTG{locations.index(i)}',icon=folium.features.CustomIcon(icon_url,icon_size=(50,50))).add_to(m)
        elif predictions_25[locations.index(i)][0]==1:
            folium.Marker(i,popup="Medium",tooltip=f'WTG{locations.index(i)}',icon=folium.features.CustomIcon(icon_url_orange,icon_size=(50,50))).add_to(m)
        else:
            folium.Marker(i,popup="Low",tooltip=f'WTG{locations.index(i)}',icon=folium.features.CustomIcon(icon_url_red,icon_size=(50,50))).add_to(m)
        
    return m

def pinpoint_WTGs(prediction = '0KW'):
    WTG_longitude = 18.97466035850479
    WTG_latitude= -69.27671861610212
    icon_url = "app/wind-turbine.png"
    coordinates = [WTG_longitude,WTG_latitude]
    m = folium.Map(location=coordinates,zoom_start=14.5, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community')
    for i in points:
        folium.Marker(i,tooltip=f'WTG{points.index(i)}',icon=folium.features.CustomIcon(icon_url,icon_size=(50,50))).add_to(m)
    return m

def app():
    st.write('''
    # Projectwind model

    This model classifies the energy level production of 25 wind turbines
    ''')

    option = st.selectbox(
        'From which wind power station would you like to predict the energy produced?',
        ('None', 'Station 1', 'Station 2'))

    if option == 'Station 1':
#        d = st.date_input(
#        "When do you want to predict",
#         datetime.datetime(2022,2,26,18,00,00))
        button = st.button('Predict')
        if button:
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.write('''
            ### The level of production of the 25 wind turbines for next 6 hours is :
            ''')
            m = pinpoint_WTGs_prediction()
            folium_static(m)
        else:
            m = pinpoint_WTGs()
            folium_static(m)
            st.write('')

        #prediction function
        #m = pinpoint_WTGs()
        #folium_static(m)

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
