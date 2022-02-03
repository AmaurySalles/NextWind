import numpy as np 
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalisation


def get_LSTM_model(n_steps_in, n_steps_out, n_features):

    model = Sequential()
    model.add(BatchNormalisation, input_shape)
    model.add(LSTM(5, activation='tanh', input_shape=(n_steps_in, n_features), return_sequences=True))
    model.add(LSTM(1, activation='tanh', return_sequences=True))
    model.add(Dense(n_steps_out, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    return model

    model.add(TimeDistributed(Dense(1)))