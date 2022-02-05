import pandas as pd
import numpy as np
import os

from projectwind.clean import clean_timesteps
from projectwind.LSTM_preproc import get_random_sequences
from projectwind.pipeline import get_pipeline


<<<<<<< HEAD
def get_samples(data, fit_data, day_length=5.5, number_of_subsamples=100, acceptable_level_of_missing_values=0.05):

    # Get data & perform  splits
    #data, fit_data, = get_data()
    data = clean_timesteps(data)
    train_data, test_data = split_test_data(data)
    X_fit, y_fit = split_fit_data(fit_data)

    # Pipeline fit
    pipeline = get_pipeline()
    pipeline.fit(X_fit)

    # Transform data & fetch sequences
    # Returns 3D array with number_of_subsamples # sequences x 25 WTG, day_length # timesteps & 5 features
    samples = get_random_sequences(train_data,
                                fitted_pipeline=pipeline,
                                day_length=day_length,
                                number_of_subsamples=number_of_subsamples,
                                acceptable_level_of_missing_values=acceptable_level_of_missing_values)

    # Shuffle WTG sequences & target
    # X_train seed=42
    # y_train seed=42

    # Save sequences for quicker upload time
    X, Y = samples
=======
>>>>>>> f0495e4f94d1883072430aa8dabe9a8ffd3140c1

def get_data(num_datasets=25):

<<<<<<< HEAD
def get_data(num_datasets):
=======
>>>>>>> f0495e4f94d1883072430aa8dabe9a8ffd3140c1
    # Take the parent dirname for the raw data
    parentdir = os.path.dirname(os.path.abspath("__file__"))
    rawdir = os.path.join(parentdir,"raw_data")

    # Output dict
    all_WTG_data = {}
    fit_data = pd.DataFrame()

    # Append all csv data files to a dict("WTG_number" : dataframe)
    for root, directory, file in os.walk(rawdir):
<<<<<<< HEAD
        for WTG_number in range (num_datasets):
            print(WTG_number+1)
=======

        for WTG_number in range (num_datasets):
            print(WTG_number+1)

>>>>>>> f0495e4f94d1883072430aa8dabe9a8ffd3140c1
            # Train/Val/Test dataset
            # Output format: Dataframe per WTG assembled in a dict("WTG_number": dataframe)
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

    # Prepare df containing scaler fit data (no need for cleaning as there are no outliers)
    # Format: timesteps concatenated / only 6 columns
    for WTG_number in all_WTG_data:
        fit_data = pd.concat((fit_data,all_WTG_data[WTG_number]),ignore_index=True)


    return all_WTG_data, fit_data


def split_test_data(data):
    data_length = len(data[0].shape)
    train_data = [data[WTG].iloc[0:-int(data_length*0.8)] for WTG in data]
    test_data = [data[WTG].iloc[-int(data_length*0.8):] for WTG in data]
    return train_data, test_data


def split_fit_data(fit_data):
    y_fit = fit_data['Power']
    X_fit = fit_data
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
