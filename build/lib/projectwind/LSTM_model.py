import numpy as np 
import pandas as pd

from tensorflow.keras import Sequential, LSTM, Dense


def get_LSTM_model(n_steps_in, n_steps_out, n_features):

    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')


    return model