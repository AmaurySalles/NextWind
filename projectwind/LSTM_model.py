import numpy as np 
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization


def init_LSTM_model(n_steps_in, n_steps_out, n_features):

    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_steps_in, n_features)))
    model.add(LSTM(16, activation='tanh', return_sequences=False))
    #model.add(LSTM(1, activation='tanh', return_sequences=False))
    model.add(Dense(n_steps_out, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    return model

