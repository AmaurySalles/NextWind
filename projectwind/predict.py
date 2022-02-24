import pandas as pd
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import tensorflow as tf

from projectwind.LSTM_weather_forecast import WindowGenerator, get_LSTM_data

train_df, val_df, test_df = get_LSTM_data(25)

def plot(x_hist, x_fc, y_pred, y_true, max_subplots=3):
        plot_col = 'Power'
        plt.figure(figsize=(12, 8))
        plot_col_index = window.column_indices[plot_col]
        max_n = min(max_subplots, len(y_true))
        for n in range(max_n):
            i = np.random.randint(0,len(y_true))
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')

            # Historical inputs
            plt.plot(window.input_indices, x_hist[i, :, plot_col_index],
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

hn_model = tf.keras.models.load_model('./checkpoint/LSTM_Forecast_Hybrid_num_3Lx16N_fc_2Lx54-18N_2x02do.h5')

X_train, X_fc_train, y_train =  window.train
X_val, X_fc_val,  y_val   =  window.val
X_test, X_fc_test,  y_test  =  window.test

y_pred = hn_model.predict([X_val, X_fc_val], batch_size=1)

plot(X_val, X_fc_val, y_pred, y_val)
