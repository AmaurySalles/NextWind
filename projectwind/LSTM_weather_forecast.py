import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from itertools import chain

from projectwind.data import get_data
from projectwind.weather import get_weather

def get_LSTM_data(num_datasets=25, freq=None):

    # Fetch csv & weather datasets
    data = get_data(num_datasets)
    
    print('### Fetching weather API data ###')
    weather = pd.read_csv('./raw_data/API_data/historical_weather_API_data.csv', index_col=0, parse_dates=True) 

    print('### Preparing datasets ###')
    train_df, val_df, test_df = list(), list(), list()
    # Data pre-processing
    for WTG_data in data:

        # Fill in na_values
        WTG_data.interpolate(axis=0, inplace=True)
        
        # Feature engineering
        WTG_data = feature_engineering(WTG_data)

        # Join with weather data
        WTG_data = pd.concat([WTG_data, weather], axis=1)
        # Slice off additional API data timestamps
        WTG_data.dropna(axis=0, inplace=True)

        # Resample on hourly basis using exponential weighted moving average (ewm)
        WTG_data = WTG_data.ewm(span=6).mean().resample('H').mean()

        # Split datasets
        n = len(WTG_data)
        train_df.append(WTG_data[0:int(n*0.7)])
        val_df.append(WTG_data[int(n*0.7):int(n*0.9)])
        test_df.append(WTG_data[int(n*0.9):])

    # Scale datasets
    train_df, val_df, test_df = scale_data(train_df, val_df, test_df)

    return train_df, val_df, test_df

def scale_data(train_df, val_df, test_df):
    
    # Find min / max of each category across all 25 WTGs (from train set only to avoid data leakage)
    scaling_data = pd.DataFrame(index=['min','max'], columns=train_df[0].columns, data=0)
    for WTG_data in train_df:
        for col in WTG_data:
            temp_min = np.min([scaling_data.loc['min', col], WTG_data[col].min(axis=0)])
            temp_max = np.max([scaling_data.loc['max', col], WTG_data[col].max(axis=0)])
            scaling_data.loc['min', col] = temp_min
            scaling_data.loc['max', col] = temp_max

    # Apply scaling to all three datasets
    pd.options.mode.chained_assignment = None
    column_names = train_df[0].columns
    for WTGs in range(len(train_df)):
        for col in column_names:
            col_min = scaling_data.loc['min',col]
            col_max = scaling_data.loc['max',col]
            # Scale each columns of each dataset
            train_df[WTGs].loc[:,col] = train_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
            val_df[WTGs].loc[:,col] = val_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
            test_df[WTGs].loc[:,col] = test_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
    pd.options.mode.chained_assignment = 'warn'
    return train_df, val_df, test_df

def feature_engineering(WTG_data):

    # Find wind direction (by correcting nacelle orientation with misalignment)
    WTG_data['Misalignment'] = WTG_data['Misalignment']* np.pi / 180 # Transform into radians
    WTG_data['Nacelle Orientation'] = WTG_data['Nacelle Orientation'] * np.pi / 180 # Transform into radians
    WTG_data['Wind_direction'] =  WTG_data['Nacelle Orientation'] - WTG_data['Misalignment']

    # Build vectors for nacelle orientation
    WTG_data['Nacelle_X'] = np.cos(WTG_data['Nacelle Orientation'])
    WTG_data['Nacelle_Y'] = np.sin(WTG_data['Nacelle Orientation'])

    # Build vectors from wind direction and wind speed
    WTG_data['Wind_X'] = WTG_data['Wind Speed'] * np.cos(WTG_data['Wind_direction'])
    WTG_data['Wind_Y'] = WTG_data['Wind Speed'] * np.sin(WTG_data['Wind_direction'])  

    # Remove superseeded columns, except wind speed
    WTG_data.drop(columns=['Misalignment','Nacelle Orientation', 'Wind_direction'], inplace=True)

    # Transform time into sin/cosine to represent periodicity
    timestamp_s = WTG_data.index.map(pd.Timestamp.timestamp)
    day = 24*60*60
    WTG_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    WTG_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    return WTG_data

def load_datasets(n_steps_in, n_steps_out):

    sequence_name = f"{n_steps_in // 6}-{n_steps_out // 6}"

    X_train = np.load(f'./projectwind/data/LSTM_sequence_X_train_{sequence_name}.npy', allow_pickle=True)
    y_train = np.load(f'./projectwind/data/LSTM_sequence_y_train_{sequence_name}.npy', allow_pickle=True)
    X_val   = np.load(f'./projectwind/data/LSTM_sequence_X_val_{sequence_name}.npy', allow_pickle=True)
    y_val   = np.load(f'./projectwind/data/LSTM_sequence_y_val_{sequence_name}.npy', allow_pickle=True)
    X_test  = np.load(f'./projectwind/data/LSTM_sequence_X_test_{sequence_name}.npy', allow_pickle=True)
    y_test  = np.load(f'./projectwind/data/LSTM_sequence_y_test_{sequence_name}.npy', allow_pickle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, 
                 input_columns=None, forecast_columns=None, label_columns=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.column_indices = {name: i for i, name in enumerate(train_df[0].columns)}

        # Work out the input column indices.
        self.input_columns = input_columns
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
        else:
            self.input_columns = train_df[0].columns
            self.input_columns_indices =  self.column_indices

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        # Work out the forecast column indices.
        if forecast_columns is not None:
            self.forecast_columns = forecast_columns
            self.forecast_columns_indices = {name: i for i, name in enumerate(forecast_columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.forecast_width = label_width
        self.label_width = label_width
        self.shift = shift
        
        # Work out window slices
        self.total_window_size = input_width + shift
        # Inputs
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        # Forecast
        self.forecast_start = self.total_window_size - self.forecast_width
        self.forecast_slice = slice(self.forecast_start, None)
        self.forecast_indices = np.arange(self.total_window_size)[self.forecast_slice]
        # Label
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input column name(s): {self.input_columns}', 
            f'Input indices: {self.input_indices}',
            f'Forecast column name(s): {self.forecast_columns}',            
            f'Forecast indices: {self.forecast_indices}',
            f'Label column name(s): {self.label_columns}',
            f'Label indices: {self.label_indices}'])


    def split_windows(self, features):
        
        # Splice correct timestamps
        inputs = features[:, self.input_slice, :]
        forecast = features[:, self.forecast_slice, :]
        labels = features[:, self.labels_slice, :]
        
        # If input, forecast & labels are specified, select requested columns
        if self.input_columns is not None:
            inputs = tf.stack([inputs[:,:, self.column_indices[name]] for name in self.input_columns],
                            axis=-1)
        inputs.set_shape([None, self.input_width, None])
            
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:, self.column_indices[name]] for name in self.label_columns],
                            axis=-1)
        labels.set_shape([None, self.label_width, None])

        # Forecast
        if self.forecast_columns is None:
            return inputs, labels
        else:
            forecast = features[:, self.forecast_slice, :]
            forecast = tf.stack([forecast[:,:, self.column_indices[name]] for name in self.forecast_columns],
                            axis=-1)
            forecast.set_shape([None, self.forecast_width, None])
            return inputs, forecast, labels

    def make_dataset(self, data):
        X_datasets = []
        X_fc_datasets = []
        y_datasets = []

        for WTG_data in data:

            # Find sequences according to window size of X and y
            WTG_data = np.array(WTG_data, dtype=np.float32)
            WTG_sequences = tf.keras.utils.timeseries_dataset_from_array(data=WTG_data,
                                                                        targets=None,
                                                                        sequence_length=self.total_window_size,
                                                                        sampling_rate=1,
                                                                        sequence_stride=self.total_window_size,
                                                                        shuffle=False,
                                                                        batch_size=32)
            # Split X and y according to window size
            WTG_sequences = WTG_sequences.map(self.split_windows)

            # Transfer from tensor to numpy array to save under .NPY format
            X_datasets.append(chain.from_iterable([X.numpy() for X, X_fc, y in WTG_sequences]))
            X_fc_datasets.append(chain.from_iterable([X_fc.numpy() for X, X_fc, y in WTG_sequences]))
            y_datasets.append(chain.from_iterable([y.numpy() for X, X_fc, y in WTG_sequences]))

        # Aggregate WTGs batches into same array
        X_array = np.array(list(chain.from_iterable(X_datasets)))
        X_fc_array = np.array(list(chain.from_iterable(X_fc_datasets)))
        y_array = np.array(list(chain.from_iterable(y_datasets)))

        X_array, X_fc_array, y_array = self.shuffle_sequences(X_array, X_fc_array, y_array)

        return X_array, X_fc_array, y_array
    
    
    # Not in current use
    # def make_dataset(self, data):

    #     # Find sequences according to window size of X and y
    #     data = np.array(data, dtype=np.float32)
    #     WTG_sequences = tf.keras.utils.timeseries_dataset_from_array(data=data,
    #                                                                 targets=None,
    #                                                                 sequence_length=self.total_window_size,
    #                                                                 sampling_rate=1,
    #                                                                 sequence_stride=self.total_window_size,
    #                                                                 shuffle=True,
    #                                                                 batch_size=32)
    #     # Split X and y according to window size
    #     WTG_sequences = WTG_sequences.map(self.split_window)

    #     return WTG_sequences
 
    def shuffle_sequences(self, X, X_fc, y, seed=42):
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(X_fc)
        np.random.seed(seed)
        np.random.shuffle(y)
        return X, X_fc, y

    def plot(self, model=None, plot_col='Power', max_subplots=3):
        inputs, forecast, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                label_col_index = 0

            plt.plot(self.label_indices, labels[n, :, label_col_index],
                        label='Labels', c='#2ca02c', marker='.')

            if model is not None:
                predictions = model([inputs, forecast])
                plt.plot(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', label='Predictions', c='#ff7f0e')

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')
            plt.tight_layout()

   
    @property
    def save_datasets(self):
        X_train, y_train = self.make_dataset(self.train_df)
        X_val, y_val = self.make_dataset(self.val_df)
        X_test, y_test = self.make_dataset(self.test_df)

        sequence_name = f"{self.input_width // 6}-{self.label_width // 6}"
        np.save(f'./projectwind/data/LSTM_sequence_X_train_{sequence_name}.npy', np.asanyarray(X_train, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_y_train_{sequence_name}.npy', np.asanyarray(y_train, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_X_val_{sequence_name}.npy', np.asanyarray(X_val, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_y_val_{sequence_name}.npy', np.asanyarray(y_val, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_X_test_{sequence_name}.npy', np.asanyarray(X_test, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_y_test_{sequence_name}.npy', np.asanyarray(y_test, dtype=object))

        return print(f"Data saved under './projectwind/data/LSTM_sequence_<dataset>_{sequence_name}.npy")

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = self.train
            # And cache it for next time
            self._example = result
        return result
