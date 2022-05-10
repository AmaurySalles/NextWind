import os
import pandas as pd
import datetime as datetime
import requests

# Period with low # NaNs, following data analysis
START_DATE = '2019-05-05'
END_DATE = '2021-09-30'

def get_WTG_data(num_datasets=25):
    """
    Fetches all data from a specified number of WTGs.
    Each WTG files contains 'Power' , 'Wind Speed', 'Misalignment', 'Rotor Speed', 'Blade Pitch' data
    ------------
    Parameters
    num_datasets: 'int' 
                  Number of WTG files to retrieve data from. Max (default) is 25
    ------------
    Returns:
    List of DataFrames, with each DataFrame containing all data from a WTG
    """
    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.abspath("__file__"))
    rawdir = os.path.join(parentdir,"raw_data/WTG_data")

    # Output dict    
    all_WTG_data = []
    print(f'### Fetching {num_datasets}xWTG data ###')
    # Append all csv data files to a dict("WTG_number" : dataframe)
    for root, directory, file in os.walk(rawdir):
        k = len(file)
        for WTG_number in range (num_datasets):
            
            # Read file
            WTG_data = pd.read_csv(root +'/' +file[WTG_number], 
                                index_col=0,
                                parse_dates=True,
                                dayfirst=True)
                       
            # Sort index in chronological order
            WTG_data.sort_index()

            # Remove duplicates
            WTG_data.drop_duplicates(inplace=True)

            # Add missing timesteps
            ref_date_range = pd.date_range(start=WTG_data.index.min(), end=WTG_data.index.max(), freq='10T')
            WTG_data = WTG_data.reindex(ref_date_range)

            # Remove start/end periods with high NaNs
            WTG_data = WTG_data.loc['2019-05-05':'2021-09-30']

            # Fill in na_values
            WTG_data.interpolate(axis=0, inplace=True)
        
            # Resample on hourly basis 
            WTG_data = WTG_data.resample('H').mean()

            # Output format: Dataframe per WTG assembled in a dict("WTG_number": dataframe)
            all_WTG_data.append(WTG_data)
            
    return all_WTG_data

def get_WTG_cat_data(category, num_datasets=25):
    """
    Fetch specific WTG data category. Useful for intial data analysis
    ------------
    Parameters
    category: 'str'
              Categories possible are: ['Power' , 'Wind Speed', 'Misalignment', 'Rotor Speed', 'Blade Pitch']
    num_datasets: 'int' 
                  Number of WTG files to retrieve data from. Max (default) is 25
    ------------
    Returns:
    Single DataFrame, each column containing data from a WTG for the category specified
    """

    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.abspath("__file__"))
    rawdir = os.path.join(parentdir,"raw_data/WTG_data")

    # Output: DataFrame with WTG category data as columns
    category_data = pd.DataFrame()
    
    print(f'### Fetching {num_datasets}xWTG {category} data ###')
    # Append all csv data files to a dict("WTG_number" : dataframe)
    for root, directory, file in os.walk(rawdir):

        for WTG_number in range (num_datasets):

            # Read file
            WTG_data = pd.read_csv(root +'/' +file[WTG_number], 
                                index_col=0,
                                parse_dates=True,
                                dayfirst=True)
            
            # Select categorical data requested
            if category in WTG_data.columns:
                WTG_data = WTG_data.pop(category)
            
            # Sort index in chronological order
            WTG_data.sort_index()

            # Remove duplicates
            WTG_data.drop_duplicates(inplace=True)

            # Add missing timesteps
            ref_date_range = pd.date_range(start=WTG_data.index.min(), end=WTG_data.index.max(), freq='10T')
            WTG_data = WTG_data.reindex(ref_date_range)

            # Remove start/end periods with high NaNs (added following data analysis)
            WTG_data = WTG_data.loc[START_DATE:END_DATE]

            # Fill in na_values
            WTG_data.interpolate(axis=0, inplace=True)
        
            # Resample on hourly basis
            WTG_data = WTG_data.resample('H').mean()

            # Append to DataFrame
            category_data[WTG_number] = WTG_data
            
    return category_data

def get_WWO_API_data(lon=19.696,lat=-71.219, start_date=START_DATE, end_date=END_DATE, frequency='1H'):
    """
    This function fetches historical wind data from World Weather Online API
    -------------
    Parameters:
    lon : 'float'
          Longitude of the location
    lat : 'float'
          Latitude of the location
    start_date: 'pd.Datetime' rounded to hours
                API request initial date
    end_date: 'pd.Datetime' rounded to hours
              API request final date
    frequency: 'str'
               Timestep frequency of the output - e.g. '1H', '10T', etc. 
               If frequency is lower than 1hr, forward fill will be applied. 
               If frequency is higher than 1hr, only matching timesteps will be returned 
               (i.e. no average over the period is performed)
    -------------
    Returns:
    A DataFrame with datetime index (from start_date to end_date) 
    and wind and gust wind speeds, and wind and gust wind directions 
    as vector coordinates, suitables for ML
    """
    
    print('### Fetching weather forecast from World Weather Online API ###')
    
    # Set requested dates into correct format
    break_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")+datetime.timedelta(hours=23)
    request_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    request_end_date = request_start_date + datetime.timedelta(days=29, hours=23)
    
    # Columns of interest from API
    col = ['windspeedKmph', 'WindGustKmph', 'winddirDegree']
    weather = pd.DataFrame(columns=col)

    # Fetch in batches of 30 days, until end date is reached
    while request_end_date <= break_date:
        
        # Avoid fetching more data than necessary (when close to end date)
        request_end_date= min(request_end_date, break_date)

        # World Weather - Historical data API url
        url = f"https://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=d4dc0a3b75ef4e749b4150417221602&q={lon},{lat}&date={request_start_date.strftime('%Y-%m-%d')}&enddate={request_end_date.strftime('%Y-%m-%d')}&tp=1&format=json"

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
    
    return weather

def get_MERRA2_data(start_date=START_DATE, end_date=END_DATE):
    """
    This function fetches historical wind data from MERRA2 satelitte (pre-downloaded)
    -------------
    Parameters:
    start_date: 'pd.Datetime' rounded to nearest 10min
                No data prior to 2019-05-05 - Select later start_date
    end_date: 'pd.Datetime' rounded to nearest 10min
              No date after 2021-09-30 - Select earlier end_date
    -------------
    Returns:
    A DataFrame with datetime index (from start_date to end_date) 
    with MERA2 wind speeds and wind direction (pre-downloaded)
    """
    print('### Fetching weather forecast data from MERRA2 ###')
    weather = pd.read_csv('./raw_data/API_data/Exported MERRA2_SCB.csv', index_col=0, parse_dates=True, dayfirst=False) 
    weather = weather.resample('H').mean()
    weather.drop(columns={'M10 [m/s]','D10 [°]','T [°C]','P [Pa]'},
                inplace=True)
    weather.rename(columns={'M50 [m/s]':'Forecast_wind_speed',
                            'D50 [°]':'Forecast_wind_direction'},
                    inplace=True)
    weather = weather.loc[start_date:end_date]
    return weather

def get_ERA5_data(start_date=START_DATE, end_date=END_DATE):
    """
    This function fetches historical wind data from ERA5 satelitte (pre-downloaded)
    -------------
    Parameters:
    start_date: 'pd.Datetime' rounded to nearest 10min
                No data prior to 2019-05-05 - Select later start_date
    end_date: 'pd.Datetime' rounded to nearest 10min
              No date after 2021-09-30 - Select earlier end_date
    -------------
    Returns:
    A DataFrame with datetime index (from start_date to end_date) 
    with ERA5 wind speeds and wind direction (pre-downloaded)
    """
    print('### Fetching weather forecast data from ERA5 ###')
    weather = pd.read_csv('./raw_data/API_data/Exported ERA5_SCB.csv', index_col=0, parse_dates=True, dayfirst=False) 
    weather = weather.resample('H').mean()
    weather.drop(columns={'M10 [m/s]','D10 [°]','T [°C]','P [hPa]'},
                inplace=True)
    weather.rename(columns={'M100 [m/s]':'Forecast_wind_speed',
                            'D100 [°]':'Forecast_wind_direction'},
                    inplace=True)
    weather = weather.loc[start_date:end_date]
    return weather


