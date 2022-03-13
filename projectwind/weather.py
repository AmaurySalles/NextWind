import numpy as np
import requests
import datetime
import pandas as pd


def get_world_weather_API_data(lon=19.696598,lat=-71.219500, start_date="01/05/19", end_date ="30/09/21", frequency='10T'):
    """
    This function fetches historical wind data from World Weather Online API
    -------------
    Parameters:
    lon : longitude of the location
    lat : latitude of the location
    start_date: API request initial date
    end_date: API request final date
    frequency: Timestep frequency of the output - e.g. '1H', '10T', etc. 
    If frequency is lower than 1hr, forward fill will be applied. 
    If frequency is higher than 1hr, only matching timesteps will be returned 
    (i.e. no average over the period is performed)
    -------------
    Returns:
    A DataFrame with datetime index (from start_date to end_date) 
    and wind and gust wind speeds, and wind and gust wind directions 
    as vector coordinates, suitables for ML
    """
    
    print('### Fetching Forecast from API ###')
    
    # Set requested dates into correct format
    break_date = datetime.datetime.strptime(end_date, "%d/%m/%y")+datetime.timedelta(hours=23)
    request_start_date = datetime.datetime.strptime(start_date, "%d/%m/%y")
    request_end_date = request_start_date + datetime.timedelta(days=29, hours=23)
    
    # Columns of interest from API
    col = ['windspeedKmph', 'WindGustKmph', 'winddirDegree']
    weather = pd.DataFrame(columns=col)

    # Fetch in batches of 30 days, until end date is reached
    while request_end_date <= break_date:
        
        # Avoid fetching more data than necessary (when close to end date)
        request_end_date= min(request_end_date, break_date)

        # World Weather - Historical data API url
        url = f"https://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=d4dc0a3b75ef4e749b4150417221602&q={lon},{lat}&date={request_start_date.strftime('%d/%m/%Y')}&enddate={request_end_date.strftime('%d/%m/%Y')}&tp=1&format=json"

        # Fetch API data for date range and location specified
        resp = requests.get(url).json()
        
        # Convert to dataframe
        df=pd.json_normalize(resp["data"]["weather"],['hourly'], errors='ignore', meta="date")
        
        # Set datetime column as index
        df.index = pd.to_datetime(df["date"]+" "+df["time"].str.zfill(4))
        
        # Upsample with forward fill, if frequency request is higher resolution than 1hr
        df = df.reindex(pd.date_range(start=request_start_date, end=request_end_date, freq=frequency),method='ffill')
        
        #Append the data to a new dataframe
        weather = pd.concat(objs=[weather, df[col]])

        # Update request dates to fetch next 30 days of data
        request_start_date = request_start_date + datetime.timedelta(days=30)
        request_end_date = request_end_date + datetime.timedelta(days=30)
    
    # Convert values to numeric - still needed?
    weather[col] = weather[col].apply(pd.to_numeric, downcast="float", errors='ignore')
    
    print('### Loaded Forecast from API ###')
    
    return feature_engineering(weather)

def get_MERRA2_data(start_date="2019-05-05", end_date="2021-09-30"):
    """
    This function fetches historical wind data from MERRA2 satelitte (pre-downloaded)
    -------------
    Parameters:
    start_date: No data prior to 2019-05-05 - Select later start_date
    end_date: No date after 2021-09-30 - Select earlier end_date
    -------------
    Returns:
    A DataFrame with datetime index (from start_date to end_date) 
    with MERA2 wind speeds and wind direction (pre-downloaded)
    """
    weather = pd.read_csv('./raw_data/API_data/Exported MERRA2_SCB.csv', index_col=0, parse_dates=True, dayfirst=False) 
    weather = weather.resample('H').mean()
    weather.drop(columns={'M10 [m/s]','D10 [°]','T [°C]','P [Pa]'},
                inplace=True)
    weather.rename(columns={'M50 [m/s]':'Wind Speed MERRA2',
                            'D50 [°]':'Wind Direction MERRA2'},
                    inplace=True)
    weather = weather.loc[start_date:end_date]
    return weather

def get_ERA5_data(start_date="2019-05-05", end_date="2021-09-30"):
    """
    This function fetches historical wind data from ERA5 satelitte (pre-downloaded)
    -------------
    Parameters:
    start_date: No data prior to 2019-05-05 - Select later start_date
    end_date: No date after 2021-09-30 - Select earlier end_date
    -------------
    Returns:
    A DataFrame with datetime index (from start_date to end_date) 
    with ERA5 wind speeds and wind direction (pre-downloaded)
    """
    weather = pd.read_csv('./raw_data/API_data/Exported ERA5_SCB.csv', index_col=0, parse_dates=True, dayfirst=False) 
    weather = weather.resample('H').mean()
    weather.drop(columns={'M10 [m/s]','D10 [°]','T [°C]','P [hPa]'},
                inplace=True)
    weather.rename(columns={'M100 [m/s]':'Wind Speed ERA5',
                            'D100 [°]':'Wind Direction ERA5'},
                    inplace=True)
    weather = weather.loc[start_date:end_date]
    return weather