from tokenize import endpats
import requests
import datetime
import pandas as pd


def get_weather(lon=19.696598,lat=-71.219500, start_date="01/05/19", end_date ="30/09/21"):

    date_1 = datetime.datetime.strptime(start_date, "%d/%m/%y")
    date_2 = date_1 + datetime.timedelta(days=29)
    break_date = datetime.datetime.strptime(end_date, "%d/%m/%y")

    col = ['Fecha','weatherCode', 'tempC', 'weatherDesc', 'precipMM', 'windspeedKmph', 'winddirDegree']
    df = pd.DataFrame(columns=col)

#Loop over all the dates to then do the GET request
    while date_2<=break_date:
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
                    new_row={
                        'Fecha' : f"{date[8:] + '-'+date[5:7] + '-'+date[:4]} {'0'+hour['time'][0]+':'+str(minute)+'0' if len(hour['time'])< 4 else hour['time'][0:2]+':'+str(minute)+'0'}",
                        'weatherCode' : hour['weatherCode'],
                        'tempC' : hour['tempC'],
                        'weatherDesc' : hour['weatherDesc'][0]['value'],
                        'precipMM' : hour['precipMM'],
                        'windspeedKmph' : hour['windspeedKmph'],
                        'winddirDegree' : hour['winddirDegree']
                    }
                    #Append the data to a new dataframe
                    df = df.append(new_row, ignore_index=True)
        date_1 = date_1 + datetime.timedelta(days=30)
        date_2 = date_2 + datetime.timedelta(days=30)
    df.set_index("timestamp", inplace=True)
    return df
