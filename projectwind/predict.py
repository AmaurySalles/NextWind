import pandas as pd
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import tensorflow as tf

from projectwind.ss.LSTM_weather_forecast import WindowGenerator, get_LSTM_data

def do_not_use():
    train_df, val_df, test_df = get_LSTM_data(25)

    n_steps_in = 3*24   # hrs
    n_steps_out = 12   # hrs

    window = WindowGenerator(input_width=n_steps_in, label_width=n_steps_out, shift=n_steps_out,
                            train_df=train_df, val_df=val_df, test_df=test_df,
                            input_columns=['Power', 'Rotor Speed', 'Wind Speed', 'Blade Pitch', 'Nacelle_X',
                                        'Nacelle_Y', 'Wind_X', 'Wind_Y'],
                            forecast_columns=['Wind Speed'],
                            label_columns=['Power'])

def plot(x_hist, x_fc, y_pred, y_true, max_subplots=3, window=None):
        # try:
        #     plot_col_index = window.column_indices[plot_col]
        #     input_col = window.input_indices[plot_col]

        
        plot_col = 'Power'
        plt.figure(figsize=(12, 8))
        plot_col_index = window.column_indices[plot_col]
        max_n = min(max_subplots, len(y_true))
        for n in range(max_n):
            i = np.random.randint(0,len(y_true))
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')

            # Historical inputs
            plt.plot(window.input_indices[plot_col], x_hist[i, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)



             # Forecast input
            if window.forecast_columns:
                 forecast_col_index = window.forecast_columns_indices.get('Wind Speed', None)
            else:
                 forecast_col_index = plot_col_index

            plt.plot(window.forecast_indices, x_fc[i, :, forecast_col_index],
                      label='Forecast Inputs', marker='o', c='blue', zorder=-10)


            # Target
            if window.label_columns:
                label_col_index = window.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                label_col_index = 0

            # Label
            plt.plot(window.label_indices, y_true[i, :, label_col_index],
                        label='Labels', c='#2ca02c', marker='.')

            # Prediction
            plt.plot(window.label_indices, y_pred[i, :],
                        marker='X', label='Predictions', c='#ff7f0e')

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')
            plt.tight_layout()


def save_X_test():
    X_test, X_fc_test,  y_test  =  window.test
    np.save(f'./projectwind/data/LSTM_sequence_X_test.npy', np.asanyarray(X_test, dtype=float))
    np.save(f'./projectwind/data/LSTM_sequence_X_fc_test.npy', np.asanyarray(X_fc_test, dtype=float))

def load_and_predict():
    hn_model = tf.keras.models.load_model('./checkpoints/Energy_model_divine-firebrand-59.h5')
    X_test = np.load('./projectwind/data/Classifier_X_test.npy', allow_pickle=True)
    X_fc_test = np.load('./projectwind/data/Classifier_X_fc_test.npy', allow_pickle=True)

    y_pred = hn_model.predict([X_test, X_fc_test], batch_size=1)
    y_pred = y_pred.round()
    return y_pred



#plot(X_val, X_fc_val, y_pred, y_val)
