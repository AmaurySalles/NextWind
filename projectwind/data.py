import pandas as pd
import numpy as np
import os

def get_data():
    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.abspath("__file__"))
    rawdir = os.path.join(parentdir,"raw_data")

    # Output dict
    all_WTG_data = {}
    
    # Append all csv data files to a dict("WTG_number" : dataframe)
    for root, directory, file in os.walk(rawdir):
        for WTG_number in range (len(file)):
            
            # File for initial analysis
            if 'raw_data.csv' in file[WTG_number]:
                pass # No longer in use
            
            # File containing scaler fit data (no need for cleaning as there are no outliers)
            # Format: timesteps concatenated / only 6 columns
            elif 'fit_data.csv' in file[WTG_number]:
                fit_data = pd.read_csv(root +'/' +file[WTG_number], 
                                        index_col=0,
                                        parse_dates=True,
                                        dayfirst=True)
                
                fit_data.rename(columns={"Desalineación Nacelle y Dirección de Viento Media 10M\n(°)": "Misalignment",
                                        "Media de Potencia Activa 10M\n(kW)": "Power",
                                        "Posición Nacelle Media 10M\n(°)":"Nacelle Orientation",
                                        "Velocidad Rotor Media 10M\n(rpm)":"Rotor Speed",
                                        "Velocidad Viento Media 10M\n(m/s)":"Wind Speed",
                                        "Ángulo Pitch Media 10M\n(°)":"Blade Pitch"},
                                        inplace=True)

            
            # Train/Val/Test dataset
            # Format: Dataframe per WTG assembled in a dict("WTG_number": dataframe)
            elif 'csv' in file[WTG_number]:
                data = pd.read_csv(root +'/' +file[WTG_number], 
                                    index_col=0,
                                    parse_dates=True,
                                    dayfirst=True)
                
                data.rename(columns={"Desalineación Nacelle y Dirección de Viento Media 10M\n(°)": "Misalignment",
                                     "Media de Potencia Activa 10M\n(kW)": "Power",
                                     "Posición Nacelle Media 10M\n(°)":"Nacelle Orientation",
                                     "Velocidad Rotor Media 10M\n(rpm)":"Rotor Speed",
                                     "Velocidad Viento Media 10M\n(m/s)":"Wind Speed",
                                     "Ángulo Pitch Media 10M\n(°)":"Blade Pitch"},
                                     inplace=True)

                all_WTG_data[WTG_number] = data
    

    return all_WTG_data, fit_data


def split_test_data(data):
    
    
    return train_data, test_data


def split_fit_data(fit_data):
    
    
    return X_fit, y_fit



def concat_fit_data():

    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.abspath("__file__"))
    rawdir = os.path.join(parentdir,"raw_data")
    print(rawdir)

    # init global DF
    globalDF = pd.read_csv(os.path.join(rawdir,"A01.csv"))
    globalDF["Turbine_id"] = 1

    # loop over the CSVs
    for i in range(2,26):
    # CSV file name
        if i <10:
            file="A0"+str(i)+".csv"
        else:
            file="A"+str(i)+".csv"
        df = pd.read_csv(os.path.join(rawdir,file))
        # add an ID for each turbine
        df["Turbine_id"] = i

        #concatenate df
        globalDF = pd.concat((globalDF,df),ignore_index=True)

        globalDF.to_csv('./raw_data/fit_data.csv')
    
    return globalDF
