import os
import pandas as pd
import numpy as np
import datetime as datetime
import requests



def get_weather(lon=19.696598,lat=-71.219500, start_date="01/05/19", end_date ="30/09/21"):
    print('### Fetching Forecast from API ###')
    date_1 = datetime.datetime.strptime(start_date, "%d/%m/%y")
    date_2 = date_1 + datetime.timedelta(days=29, hours=23)
    break_date = datetime.datetime.strptime(end_date, "%d/%m/%y")+datetime.timedelta(hours=23)

    col = ['windspeedKmph', 'WindGustKmph', 'winddirDegree']
    weather = pd.DataFrame(columns=col)

    #Loop over all the dates to then do the GET request
    while date_2<=break_date:
        date_2= min(date_2, break_date)
        # print(date_1, date_2)
        url = f"https://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=d4dc0a3b75ef4e749b4150417221602&q={lon},{lat}&date={date_1.strftime('%d/%m/%Y')}&enddate={date_2.strftime('%d/%m/%Y')}&tp=1&format=json"

        #Get the json for a specific range of dates and for specific zone
        resp = requests.get(url).json()
        #Create a dataframe
        df=pd.json_normalize(resp["data"]["weather"],['hourly'], errors='ignore', meta="date")
        #fill with 0 the hour and create the Fecha column
        df["Fecha"] = pd.to_datetime(df["date"]+" "+df["time"].str.zfill(4))
        # Set Fecha as index
        df.set_index("Fecha", inplace=True)
        #add the minutes to the DF
        df = df.reindex(pd.date_range(start=date_1, end=date_2, freq="10min"),method='ffill')
        #Append the data to a new dataframe
        
        weather = pd.concat(objs=[weather, df[['windspeedKmph', 'WindGustKmph', 'winddirDegree']]])
        date_1 = date_1 + datetime.timedelta(days=30)
        date_2 = date_2 + datetime.timedelta(days=30)
        

    weather[['windspeedKmph', 'WindGustKmph', 'winddirDegree']] = weather[['windspeedKmph', 'WindGustKmph', 'winddirDegree']].apply(pd.to_numeric, downcast="float", errors='ignore')
    # print(weather)
    # print(weather.info())
    print('### Loaded Forecast from API ###')
    weather = feature_engineering(weather)
    weather.to_csv('./raw_data/API_data/historical_weather_API_data.csv')
    return weather

def feature_engineering(df):

    df['windSpeed_API'] = df['windspeedKmph'] * 1000 / (60*60) # Transform into m/s
    df['windGust_API'] = df['WindGustKmph'] * 1000 / (60*60) # Transform into m/s
    df['windDirDegree'] = df['winddirDegree']* np.pi / 180 # Transform into radians

    # Build vectors from wind direction and wind speed
    df['Wind_API_X'] = df['windSpeed_API'] * np.cos(df['windDirDegree'])
    df['Wind_API_Y'] = df['windSpeed_API'] * np.sin(df['windDirDegree'])

    # Build vectors from wind direction and wind gust
    df['WindGust_API_X'] = df['windGust_API'] * np.cos(df['windDirDegree'])
    df['WindGust_API_Y'] = df['windGust_API'] * np.sin(df['windDirDegree'])

    # Remove superseeded columns, except wind speed
    df.drop(columns=['windDirDegree', 'windspeedKmph', 'WindGustKmph', 'winddirDegree'], inplace=True)

    return df