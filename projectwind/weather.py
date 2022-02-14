from tokenize import endpats
import requests
import datetime
import pandas as pd
import numpy as np


def get_weather(lon=19.696598,lat=-71.219500, start_date="01/05/19", end_date ="30/09/21"):

    date_1 = datetime.datetime.strptime(start_date, "%d/%m/%y")
    date_2 = date_1 + datetime.timedelta(days=29)
    break_date = datetime.datetime.strptime(end_date, "%d/%m/%y")

    col = ['Fecha','weatherCode', 'tempC', 'weatherDesc', 'precipMM', 'windspeedKmph', 'winddirDegree']
    df = pd.DataFrame(columns=col)

    #Loop over all the dates to then do the GET request
    counter = 0
    while counter<=3:
        print(counter)
        date_2= min(date_2, break_date)
        print(date_1, date_2)
        url = f"https://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=32c9c3f028054024b1c160353220402&q={lon},{lat}&date={date_1.strftime('%d/%m/%Y')}&enddate={date_2.strftime('%d/%m/%Y')}&tp=1&format=json"

        #Get the json for a specific range of dates and for specific zone
        resp = requests.get(url).json()
        #Loop over every day in the range
        for day in resp['data']['weather']:
            date = day['date']
            #Loop over every hour
            for hour in day['hourly']:
                for minute in range(0,6):

                    #Create a new row for each day hour
                    #28/09/21 13:20
                    # new_row={
                    #     'Fecha' : f"{ date[:4]+ '-'+date[5:7] + '-'+date[8:]} {'0'+hour['time'][0]+':'+str(minute)+'0:00' if len(hour['time'])< 4 else hour['time'][0:2]+':'+str(minute)+'0:00'}",
                    #     'weatherCode' : int(hour['weatherCode']),
                    #     'tempC' : int(hour['tempC']),
                    #     'weatherDesc' : hour['weatherDesc'][0]['value'],
                    #     'precipMM' : hour['precipMM'],
                    #     'windspeedKmph' : float(hour['windspeedKmph']),
                    #     'winddirDegree' : float(hour['winddirDegree'])
                    # }
                    new_row = pd.DataFrame(columns=col)
                    new_row['Fecha'] = f"{ date[:4]+ '-'+date[5:7] + '-'+date[8:]} {'0'+hour['time'][0]+':'+str(minute)+'0:00' if len(hour['time'])< 4 else hour['time'][0:2]+':'+str(minute)+'0:00'}"
                    new_row['weatherCode'] = int(hour['weatherCode'])
                    new_row['tempC'] = int(hour['tempC'])
                    new_row['weatherDesc'] = hour['weatherDesc'][0]['value']
                    new_row['precipMM'] = hour['precipMM']
                    new_row['windspeedKmph'] = float(hour['windspeedKmph'])
                    new_row['winddirDegree'] = float(hour['winddirDegree'])
                    print(new_row)
                    #Append the data to a new dataframe
                    df = pd.concat([df, new_row], axis=0, ignore_index=True)
        date_1 = date_1 + datetime.timedelta(days=30)
        date_2 = date_2 + datetime.timedelta(days=30)
        counter += 1
    df.Fecha=pd.to_datetime(df.Fecha)
    df.set_index("Fecha", inplace=True)
    df.index= df.index- pd.DateOffset(hours=12)
    return feature_engineering(df)

def feature_engineering(df):

    # Find wind direction (by correcting nacelle orientation with windspeedKmph)
    df['winddirDegree'] = df['winddirDegree']* np.pi / 180 # Transform into radians


    # Build vectors from wind direction and wind speed
    df['Wind_API_X'] = df['windspeedKmph'] * np.cos(df['winddirDegree'])
    df['Wind_API_Y'] = df['windspeedKmph'] * np.sin(df['winddirDegree'])

    # Remove superseeded columns, except wind speed
    df.drop(columns=['windspeedKmph','winddirDegree'], inplace=True)

    return df
