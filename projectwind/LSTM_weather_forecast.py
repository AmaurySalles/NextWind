import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import chain

from projectwind.data import get_data
from projectwind.weather import get_weather

def get_LSTM_data(num_datasets=25, period=None):

    # Fetch csv & weather datasets
    data = get_data(num_datasets)
    
    print('### Fetching weather API data ###')
    weather = pd.read_csv('./raw_data/API_data/Exported ERA5_SCB.csv', index_col=0, parse_dates=True, dayfirst=False) 
    weather = weather.resample('H').mean()
    
    
    # Data pre-processing
    print('### Preparing datasets ###')
    train_df, val_df, test_df = list(), list(), list()
    
    for WTG_data in data:

        # Fill in na_values
        WTG_data.interpolate(axis=0, inplace=True)
        
        # Resample on hourly basis using exponential weighted moving average (ewm)
        WTG_data = WTG_data.ewm(span=6).mean().resample('H').mean()

        # Join with weather data
        WTG_data = pd.concat([WTG_data, weather[['M100 [m/s]','D100 [°]']]], axis=1)
        # Slice off additional API data timestamps
        WTG_data.dropna(axis=0, inplace=True)

        # Feature engineering
        WTG_data = WTG_feature_engineering(WTG_data)
       
        # Resampling to smooth out curves
        if period is not None:
            WTG_data = WTG_data.resample(period).mean()

        # Split datasets
        n = len(WTG_data)
        train_df.append(WTG_data[0:int(n*0.7)])
        val_df.append(WTG_data[int(n*0.7):int(n*0.9)])
        test_df.append(WTG_data[int(n*0.9):])

    # Scale datasets
    train_df, val_df, test_df = min_max_scale_data(train_df, val_df, test_df)

    return train_df, val_df, test_df

def std_scale_data(train_df, val_df, test_df):
    
    # Apply scaling to all three datasets
    pd.options.mode.chained_assignment = None
    column_names = train_df[0].columns
    for WTGs in range(len(train_df)):
        for col in column_names:
            if ('Target' in col) or ('_' in col): # _ represents the X & Y direction vectors (already between 1 and -1)
                print(col)
                pass
            else:
                col_std =   train_df[WTGs][col].std()
                col_mean =  train_df[WTGs][col].mean()
                # Scale each columns of each dataset
                train_df[WTGs].loc[:,col] = train_df[WTGs][col].apply(lambda x: (x - col_mean) / col_std)
                val_df[WTGs].loc[:,col] = val_df[WTGs][col].apply(lambda x: (x - col_mean) / col_std)
                test_df[WTGs].loc[:,col] = test_df[WTGs][col].apply(lambda x: (x - col_mean) / col_std)
    pd.options.mode.chained_assignment = 'warn'
    return train_df, val_df, test_df

def min_max_scale_data(train_df, val_df, test_df):
    
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
            if ('Power' in col) or ('_' in col): # _ represents the X & Y direction vectors (already between 1 and -1)
                pass
            else:
                col_min = scaling_data.loc['min',col]
                col_max = scaling_data.loc['max',col]
                # Scale each columns of each dataset
                train_df[WTGs].loc[:,col] = train_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
                val_df[WTGs].loc[:,col] = val_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
                test_df[WTGs].loc[:,col] = test_df[WTGs][col].apply(lambda x: (x - col_min) / (col_max - col_min))
    pd.options.mode.chained_assignment = 'warn'
    return train_df, val_df, test_df

def WTG_feature_engineering(df):

    # Find wind direction (by correcting nacelle orientation with misalignment)
    df['Misalignment'] = df['Misalignment']* np.pi / 180 # Transform into radians
    df['Nacelle Orientation'] = df['Nacelle Orientation'] * np.pi / 180 # Transform into radians
    df['Wind_direction'] =  df['Nacelle Orientation'] - df['Misalignment']

    # Build vectors for nacelle orientation
    df['Nacelle_X'] = np.cos(df['Nacelle Orientation'])
    df['Nacelle_Y'] = np.sin(df['Nacelle Orientation'])

    # Build vectors from wind direction
    df['Wind_X'] = np.cos(df['Wind_direction']) 
    df['Wind_Y'] = np.sin(df['Wind_direction'])  

    # Build vectors from wind direction forecast
    df['D100 [°]'] = df['D100 [°]'] * np.pi / 182 # Transform into radians
    df['MERA2_X'] = np.cos(df['D100 [°]'])
    df['MERA2_Y'] = np.sin(df['D100 [°]'])

    # Remove superseeded columns, except wind speed
    df.drop(columns=['Misalignment','Nacelle Orientation', 'Wind_direction', 'D100 [°]'], inplace=True)

    return df

def WWO_feature_engineering(df):

    df['windSpeed_API'] = df['windspeedKmph'] * 1000 / (60*60) # Transform into m/s
    df['windGust_API'] = df['WindGustKmph'] * 1000 / (60*60) # Transform into m/s
    df['windDirDegree'] = df['winddirDegree']* np.pi / 180 # Transform into radians

    # Build vectors from wind direction and wind speed
    df['Wind_API_X'] = df['windSpeed_API'] * np.cos(df['windDirDegree'])
    df['Wind_API_Y'] = df['windSpeed_API'] * np.sin(df['windDirDegree'])

    # Build vectors from wind direction and wind gust
    df['WindGust_API_X'] = df['windGust_API'] * np.cos(df['windDirDegree'])
    df['WindGust_API_Y'] = df['windGust_API'] * np.sin(df['windDirDegree'])

    # Remove superseeded columns, except wind speed
    df.drop(columns=['windDirDegree', 'windspeedKmph', 'WindGustKmph', 'winddirDegree'], inplace=True)

    return df

def load_sequences(n_steps_in, n_steps_out):

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
        self.forecast_columns = forecast_columns
        if self.forecast_columns is not None:
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
        
        input_details = '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input column name(s): {self.input_columns}', 
            f'Input indices: {self.input_indices}'])

        label_details = '\n'.join([  
            f'Label column name(s): {self.label_columns}', 
            f'Label indices: {self.label_indices}'])

        if self.forecast_columns is not None:
            forecast_details = '\n'.join([f'Forecast column name(s): {self.forecast_columns}',            
                                          f'Forecast indices: {self.forecast_indices}'])
            _repr = input_details + '\n' + forecast_details + '\n' + label_details
        else:
            _repr = input_details + '\n' + label_details
        
        return _repr

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

    def make_sequences(self, data, classification=False):
        X_datasets = []
        X_fc_datasets = []
        y_datasets = []
        energy_datasets = []

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

            if self.forecast_columns is None:
                # Transfer from tensor to numpy array to save under .NPY format
                X_datasets.append(chain.from_iterable([X.numpy() for X, y in WTG_sequences]))
                y_datasets.append(chain.from_iterable([y.numpy() for X, y in WTG_sequences]))
            else:
                # Transfer from tensor to numpy array to save under .NPY format
                X_datasets.append(chain.from_iterable([X.numpy() for X, X_fc, y in WTG_sequences]))
                X_fc_datasets.append(chain.from_iterable([X_fc.numpy() for X, X_fc, y in WTG_sequences]))
                y_datasets.append(chain.from_iterable([y.numpy() for X, X_fc, y in WTG_sequences]))
            
            if classification == True:
                # Sum target power values into a single energy value
                seq_energy = self.classify_target(y_datasets)
                energy_datasets.append(seq_energy)

        # Aggregate batches into one array (batch generator done through fit function)
        if self.forecast_columns is None:
            X_array = np.array(list(chain.from_iterable(X_datasets)))
            if classification == True:
                y_array = np.array(list(chain.from_iterable(energy_datasets)))
            else:
                y_array = np.array(list(chain.from_iterable(y_datasets)))
            
            # Shuffle sequences
            X_array = self.shuffle_sequences(X_array)
            y_array = self.shuffle_sequences(y_array)

            return X_array, y_array
        
        else:
            X_array = np.array(list(chain.from_iterable(X_datasets)))
            X_fc_array = np.array(list(chain.from_iterable(X_fc_datasets)))
            if classification == True:
                y_array = np.array(list(chain.from_iterable(energy_datasets)))
            else:
                y_array = np.array(list(chain.from_iterable(y_datasets)))

            # Shuffle sequences
            X_array = self.shuffle_sequences(X_array)
            X_fc_array = self.shuffle_sequences(X_fc_array)
            y_array = self.shuffle_sequences(y_array)

            return X_array, X_fc_array, y_array

    def make_sequences_with_forecast(self, data):
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

        # X_array = self.shuffle_sequences(X_array)
        # X_fc_array = self.shuffle_sequences(X_fc_array)
        # y_array = self.shuffle_sequences(y_array)

        return X_array, X_fc_array, y_array
    
    
    # Not in current use
    # def make_sequences(self, data):

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

    def classify_target(target_list):
        for batch in target_list:
                seq_energy = []
                for seq in batch:
                    total = 0
                    for i in seq:
                        total += i[0]
                    seq_energy.append(total)
        return seq_energy
 
    def shuffle_sequences(self, data, seed=42):
        np.random.seed(seed)
        np.random.shuffle(data)
        return data

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
                if self.forecast_columns is None:
                    predictions = model(inputs)
                else:
                    predictions = model([inputs, forecast])
                    plt.plot(self.label_indices, predictions[n, :, label_col_index],
                                marker='X', label='Predictions', c='#ff7f0e')

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')
            plt.tight_layout()

   
    @property
    def save_datasets(self):
        X_train, y_train = self.make_sequences(self.train_df)
        X_val, y_val = self.make_sequences(self.val_df)
        X_test, y_test = self.make_sequences(self.test_df)

        sequence_name = f"{self.input_width // 6}-{self.label_width // 6}"
        np.save(f'./projectwind/data/LSTM_sequence_X_train_{sequence_name}.npy', np.asanyarray(X_train, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_y_train_{sequence_name}.npy', np.asanyarray(y_train, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_X_val_{sequence_name}.npy', np.asanyarray(X_val, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_y_val_{sequence_name}.npy', np.asanyarray(y_val, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_X_test_{sequence_name}.npy', np.asanyarray(X_test, dtype=object))
        np.save(f'./projectwind/data/LSTM_sequence_y_test_{sequence_name}.npy', np.asanyarray(y_test, dtype=object))

        return print(f"Data saved under './projectwind/data/LSTM_sequence_<dataset>_{sequence_name}.npy")

    @property
    def train_sequences(self, classification=False, load=False):
        if self.forecast_columns is not None:
            return self.make_sequences_with_forecast(self.train_df)
        else:
            return self.make_sequences(self.train_df)

    @property
    def val_sequences(self, classification=False, load=False):
        if self.forecast_columns is not None:
            return self.make_sequences_with_forecast(self.val_df)
        else:
            return self.make_sequences(self.val_df)

    @property
    def test_sequences(self, classification=False, load=False):
            return self.load_or_make_sequences(self.test_df)

    def load_or_make_sequences(self, dataset, classification=False, load=False):
        
        if load == True:
            if self.forecast_columns is not None:
                return self.load_sequences_with_forecast(self.test_df, self.input_width, self.label_width)
            else:
                return self.make_sequences(self.test_df)
                return self.load_sequences_with_forecast(self.window)
        if self.forecast_columns is not None:
            return self.make_sequences_with_forecast(self.test_df)
        else:
            return self.make_sequences(self.test_df)


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
