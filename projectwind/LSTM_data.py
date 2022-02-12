from operator import ge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import chain

from projectwind.data import get_data
from projectwind.weather import get_weather

def get_LSTM_data(num_datasets=25):

    # Fetch csv dataset
    data = get_data(num_datasets)
    weather = get_weather()
    print(weather)
    train_df, val_df, test_df = list(), list(), list()


    # Data pre-processing
    for WTG_data in data:

        # Fill in na_values
        WTG_data.interpolate(axis=0, inplace=True)
        print(WTG_data)
        # Join with weather data
        WTG_data = pd.concat([WTG_data, weather], axis=1)
        # Feature engineering
        WTG_data = feature_engineering(WTG_data)

        # # Split datasets
        n = len(WTG_data)
        train_df.append(WTG_data[0:int(n*0.7)])
        val_df.append(WTG_data[int(n*0.7):int(n*0.9)])
        test_df.append(WTG_data[int(n*0.9):])

    return train_df, val_df, test_df


def feature_engineering(WTG_data):

    # Find wind direction (by correcting nacelle orientation with misalignment)
    WTG_data['Misalignment'] = WTG_data['Misalignment']* np.pi / 180 # Transform into radians
    WTG_data['Nacelle Orientation'] = WTG_data['Nacelle Orientation'] * np.pi / 180 # Transform into radians
    WTG_data['Wind_direction'] =  WTG_data['Nacelle Orientation'] - WTG_data['Misalignment']



    # Build vectors from wind direction and wind speed
    WTG_data['Wind_X'] = WTG_data['Wind Speed'] * np.cos(WTG_data['Wind_direction'])
    WTG_data['Wind_Y'] = WTG_data['Wind Speed'] * np.sin(WTG_data['Wind_direction'])

    # Build vectors for nacelle orientation
    WTG_data['Nacelle_X'] = np.cos(WTG_data['Nacelle Orientation'])
    WTG_data['Nacelle_Y'] = np.sin(WTG_data['Nacelle Orientation'])

    # Remove superseeded columns, except wind speed
    WTG_data.drop(columns=['Misalignment','Nacelle Orientation', 'Wind_direction'], inplace=True)

    # Transform time into sin/cosine to represent periodicity
    timestamp_s = WTG_data.index.map(pd.Timestamp.timestamp)
    day = 24*60*60
    WTG_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    WTG_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    return WTG_data

def define_window(n_steps_in, n_steps_out, train_df, val_df, test_df):

    window = WindowGenerator(label_columns=['Power'],
                         input_width=n_steps_in, label_width=n_steps_out, shift=n_steps_out,
                         train_df=train_df, val_df=val_df, test_df=test_df)

    return window

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
                 label_columns=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


    def split_window(self, features):
        inputs = features[:,self.input_slice, :]
        labels = features[:,self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:, self.column_indices[name]] for name in self.label_columns],
                              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def plot(self, model=None, plot_col='Power', max_subplots=3):
        inputs, labels = self.example
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
                continue

            plt.plot(self.label_indices, labels[n, :, label_col_index],
                        label='Labels', c='#2ca02c', marker='.')

            if model is not None:
                predictions = model(inputs)
                plt.plot(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', label='Predictions', c='#ff7f0e')

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')
            plt.tight_layout()


    def make_dataset(self, data):

        # Find sequences according to window size of X and y
        data = np.array(data, dtype=np.float32)
        WTG_sequences = tf.keras.utils.timeseries_dataset_from_array(data=data,
                                                                    targets=None,
                                                                    sequence_length=self.total_window_size,
                                                                    sampling_rate=1,
                                                                    sequence_stride=self.total_window_size,
                                                                    shuffle=True,
                                                                    batch_size=32)
        # Split X and y according to window size
        WTG_sequences = WTG_sequences.map(self.split_window)

        return WTG_sequences

    # Not in current use
    def make_dataset_vAll(self, data):
        # X_datasets = []
        # y_datasets = []

        datasets = []
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
            WTG_sequences = WTG_sequences.map(self.split_window)

            # # Transfer from tensor to numpy array to save under .NPY format
            # X_datasets.append(chain.from_iterable([X.numpy() for X, y in WTG_sequences]))
            # y_datasets.append(chain.from_iterable([y.numpy() for X, y in WTG_sequences]))
            # X_datasets.append([X for X, y in WTG_sequences])
            # y_datasets.append([y for X, y in WTG_sequences])
            datasets.append(WTG_sequences)
        # Aggregate WTGs batches into same array
        # X_array = np.array(list(chain.from_iterable(X_datasets)))
        # y_array = np.array(list(chain.from_iterable(y_datasets)))
        array = np.array(list(chain.from_iterable(datasets)))
        # Shuffle the array to mix WTGs and sequences
        #X_array, y_array = self.shuffle_sequences(X_array, y_array)

        return array

    def shuffle_sequences(self, X, y, seed=42):
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        return X, y


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

        return X_train, y_train, X_val, y_val, X_test, y_test

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
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
