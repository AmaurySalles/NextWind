import pandas as pd
import numpy as np
import os

def concatenate_df():
    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    rawdir = os.path.join(parentdir,"raw_data")

    # init global DF
    globalDF = pd.DataFrame()
    
    # Append all csv files with correct suffix to dataframe
    global_df = pd.DataFrame()

    for root, directory, file in os.walk(rawdir):
        for WTG_number in range (len(file)):
            if 'csv' in file[WTG_number]:
                temp = pd.read_csv(root +'/' +file[WTG_number], 
                                    index_col=0,
                                    parse_dates=True,
                                    dayfirst=True)
                for col in temp:
                    if 'Desalineación' in col:
                        global_df[f'Misalignment {WTG_number+1}'] = temp[col].copy()
                    elif 'Potencia Activa' in col:
                        global_df[f'Power {WTG_number+1}'] = temp[col].copy()
                    elif 'Posición Nacelle' in col:
                        global_df[f'Nacelle Orientation {WTG_number+1}'] = temp[col].copy()
                    elif 'Velocidad Rotorelle' in col:
                        global_df[f'Rotor Speed {WTG_number+1}'] = temp[col].copy()
                    elif 'Velocidad Viento' in col:
                        global_df[f'Wind Speed {WTG_number+1}'] = temp[col].copy()
                    elif 'Pitch' in col:
                        global_df[f'Blade Pitch {WTG_number+1}'] = temp[col].copy()

global_df.to_csv('../projectwind/data/raw_data.csv')

            


if __name__ =="__main__":
    df =concatenate_df()
    print(df.head())
