import pandas as pd
import numpy as np

def add_timestamps(raw_data):
    results = {}
    for file_number in range(5):
        df = raw_data[file_number]
        ref_date_range = pd.date_range(start='1/05/2019', end='30/09/2021',freq='10T')
        ref_df = pd.DataFrame(index=ref_date_range)
        clean_data = df.reindex(ref_df.index)
        results[file_number] = clean_data
    return results

# def fill_na_with_mean(column):
#     column.fillna('', inplace=True)
#     for i in range(len(column)):
#           if column[i] == '' :
#                 column[i] = (column[i-1] + column[i+1])/2
#     return column
