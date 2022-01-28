import pandas as pd
import numpy as np

def nacelle():


def clean_timestamps():
    directory_path = "../raw_data/drive-download-20220127T182127Z-001/"
    file_names = ['A01.csv','A02.csv','A03.csv','A04.csv','A05.csv','A06.csv','A07.csv','A08.csv','A09.csv','A10.csv','A11.csv','A12.csv','A13.csv','A14.csv','A15.csv','A16.csv','A17.csv','A18.csv','A19.csv','A20.csv','A21.csv','A22.csv','A23.csv','A24.csv','A25.csv']
    results = {}
    for file in file_names:

        df = pd.read_csv(directory_path+file)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df.set_index('Fecha',inplace=True)
        ref_date_range = pd.date_range(start='1/05/2019', end='30/09/2021',freq='10T')
        ref_df = pd.DataFrame(index=ref_date_range)
        clean_data = df.reindex(ref_df.index)
        new_df = pd.merge(ref_df,clean_data,left_index=True, right_index=True,how='outer')
        results[file] = new_df

    return results
