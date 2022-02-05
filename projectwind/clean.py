import pandas as pd

# Clean data
# Sort index in chronological order
# Remove duplicates & add missing timestamps
# Remove start/end periods with high NaNs

def clean_timesteps(data):
    """
    Cleans data.index
    # Sort index in chronological order
    # Remove duplicates & add missing timestamps
    # Remove start/end periods with high NaNs
    Returns list of pd.DataFrames
    """
    results = []
    for WTG_data in data.values():

        # Sort index in chronological order
        WTG_data.sort_index()

        # Remove duplicates
        WTG_data.drop_duplicates(inplace=True)

        # Add missing timesteps
        ref_date_range = pd.date_range(start=WTG_data.index.min(), end=WTG_data.index.max(), freq='10T')
        WTG_data = WTG_data.reindex(ref_date_range)

        # Remove start/end periods with high NaNs
        WTG_data = WTG_data.loc['2019-05-05':'2021-09-30']
        
        # Return results
        results.append(WTG_data)
    
    return results

def clean_LSTM_data(data):
    """
    Cleans data.values
    # Interpolates each colum along its index (chronologically)
    """
    for WTG_data in data:
        WTG_data.interpolate(axis=0, inplace=True)
    
    return data


# def fill_na_with_mean(column):
#     column.fillna('', inplace=True)
#     for i in range(len(column)):
#           if column[i] == '' :
#                 column[i] = (column[i-1] + column[i+1])/2
#     return column
