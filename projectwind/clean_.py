import pandas as pd
import numpy as np

def add_timestamps():
    results = {}
    for file in dataframes.keys():
        df = dataframes[file]
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df.set_index('Fecha',inplace=True)
        ref_date_range = pd.date_range(start='1/05/2019', end='30/09/2021',freq='10T')
        ref_df = pd.DataFrame(index=ref_date_range)
        clean_data = df.reindex(ref_df.index)
        new_df = pd.merge(ref_df,clean_data,left_index=True, right_index=True,how='outer')
        results[file] = new_df
    results['A02.csv'].drop(columns='Unnamed: 7',inplace= True)
    return results

def fill_na_with_mean(column):
    column.fillna('', inplace=True)
    for i in range(len(column)):
          if column[i] == '' :
                column[i] = (column[i-1] + column[i+1])/2
    return column
