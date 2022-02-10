import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from projectwind.data import get_data

# LSTM Preprocessing steps:

def get_LSTM_data(num_datasets=25):

    # Fetch csv dataset
    data = get_data(num_datasets)

    train_df, val_df, test_df = list(), list(), list()

    # Data pre-processing
    for WTG_data in data:
        
        # Fill in na_values
        WTG_data.interpolate(axis=0, inplace=True)
        
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
    WTG_data['Misalignment'] = WTG_data['Misalignment'].apply(lambda x: x if x <=180 else (360 - x)*-1)
    WTG_data['Wind_direction'] =  WTG_data['Nacelle Orientation'] - WTG_data['Misalignment']
    
    # Build vectors from wind direction and wind speed
    WTG_data['Wind_direction'] = WTG_data['Wind_direction'] * np.pi / 180 # Transform into radians
    WTG_data['Wind_X'] = WTG_data['Wind Speed'] * np.cos(WTG_data['Wind_direction'])
    WTG_data['Wind_Y'] = WTG_data['Wind Speed'] * np.sin(WTG_data['Wind_direction'])
    WTG_data.drop(columns=['Misalignment','Nacelle Orientation'], inplace=True)

    # Transform time into sin/cosine to represent periodicity  
    timestamp_s = WTG_data.index.map(pd.Timestamp.timestamp)
    day = 24*60*60
    WTG_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    WTG_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    return WTG_data


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
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],
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

    
    def make_dataset(self, data):
        
        data = np.array(data, dtype=np.float32)
        dataset = tf.keras.utils.timeseries_dataset_from_array(data=data,
                                                            targets=None,
                                                            sequence_length=self.total_window_size,
                                                            sampling_rate=1,
                                                            sequence_stride=1,
                                                            shuffle=True,
                                                            batch_size=32)

        dataset = dataset.map(self.split_window)

        return dataset
    
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
