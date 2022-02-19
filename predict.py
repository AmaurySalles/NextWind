import os
from math import sqrt

import joblib
from projectwind.LSTM_data import get_LSTM_data, WindowGenerator
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf


train_df, val_df, test_df = get_LSTM_data()

# Create window
n_steps_in = 3 * 24   # hrs
n_steps_out = 6   # hrs
window = WindowGenerator(input_width=n_steps_in, label_width=n_steps_out, shift=n_steps_out,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                         forecast_columns=['windSpeed_API','windGust_API',
                                        'Wind_API_X', 'Wind_API_Y',
                                        'WindGust_API_X','WindGust_API_Y'],
                         label_columns=['Power'])

# Fetch test data
X_test,  X_fc_test,  y_test  =  window.test

#Load model
model = tf.keras.models.load_model('/checkpoint/LSTM_Forecast_Hybrid_num_4Lx16N+18d_fc_2Lx54-18N_3x02do.h5')

y_pred = model.predict([X_test, X_fc_test], batch_size=1)

def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        model,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res



if __name__ == '__main__':
