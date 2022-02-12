from cgi import test
import pandas as pd
import numpy as np

from projectwind.data import get_data
from projectwind.clean import clean_timesteps
# LSTM Preprocessing steps:

# Impute missing values (mean)
# Split into sequences (chronological order)
# Retrieve target from samples
# Split train, val & test sets
# Save samples
# Load samples

def init_LSTM_data(num_datasets=1, day_length=5.5):

    # Get data & perform  splits
    data = get_data(num_datasets) # returns dict
    data = clean_timesteps(data)  # returns list[WTG_df]
    data = clean_LSTM_data(data)  # returns list[WTG_df]
    datasets = split_train_val_test_split(data, day_length) # returns dict of list train[WTG_df], val[WTG_df], test[WTG_df] (datasets)

    # Transform data & fetch sequences
    # Returns 3D array with number_of_subsamples # sequences x 25 WTG, day_length # timesteps & 5 features
    cleaned_datasets = dict()
    for name, data in datasets.items():  # Datasets being 'train_data', 'val_data', 'test_data', each containing a list of WTG_df
        samples = get_sequences(data, day_length)
        X, Y = extract_target_from_sequences(samples, 0.5)

        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        print(X.shape)
        print(Y.shape)

        # Shuffle WTG sequences & target
            # X_train seed=42
            # y_train seed=42

        # Save sequences for quicker upload time
        np.save(f'./projectwind/data/LSTM_sequence_{name}_X_samples.npy', X)
        np.save(f'./projectwind/data/LSTM_sequence_{name}_y_samples.npy', Y)

        cleaned_datasets[name] = (X, Y)

    return cleaned_datasets['train'][0], cleaned_datasets['train'][1], cleaned_datasets['val'][0], cleaned_datasets['val'][1], cleaned_datasets['test'][0], cleaned_datasets['test'][1]

def clean_LSTM_data(data):
    """
    Cleans data.values
    # Interpolates each colum along its index (chronologically)
    """
    for WTG_data in data:
        WTG_data.interpolate(axis=0, inplace=True)

    return data


def split_train_val_test_split(data, day_length):

    # Find index split (per turbine)
    seq_len = int(24 * 6 * day_length) # Length of sequence
    seq_num = len(data[0]) // (720+72) # Find number of seq possible per turbine

    test_idx_start = int(seq_num * (0.8 * seq_len))
    val_idx_start = int(seq_num * (0.6 * seq_len))

    train_data, val_data, test_data = list(), list(), list()

    for WTG_data in data:
        train_data.append(WTG_data.iloc[0:val_idx_start])
        val_data.append(WTG_data.iloc[val_idx_start:test_idx_start])
        test_data.append(WTG_data.iloc[test_idx_start:])

    datasets = dict(train=train_data, val=val_data, test=test_data)

    return datasets


def get_sequences(data, day_length):
    """
    Given a dict of "WTG : dataframes" `data`, return a list of sequences of number_of_subsamples and of `day_length` index, with % of acceptable_missing_values.
    """
    seq_len = int(24*6*day_length)
    seq_num = len(data[0]) // seq_len
    sequence_list = []
    for WTG_data in data:

        # Slice samples chronologically
        idx_start = 0
        for i in range(seq_num):
            idx_end = idx_start + seq_len
            seq = WTG_data.iloc[idx_start:idx_end]
            sequence_list.append(seq)
            idx_start = idx_end
    print("seq_num per WTG:", len(sequence_list)//len(data))
    return sequence_list


def extract_target_from_sequences(sequence_list, target_day_length):
    '''
    Create one single random (X,y) array pair for each data sequence.
    '''
    sequence_length = sequence_list[0].shape[0]
    target_length = int(target_day_length * 6 * 24)  # convert from days to 10min periods

    Y, X = [], []
    for seq in sequence_list:
        print(seq)
        # Get target
        y_sample = seq['Power'].iloc[seq.shape[0]-target_length:]
        y_sample = y_sample.interpolate()
        Y.append(np.array(y_sample))

        # Remove target timestamps
        X_sample = seq[0:(sequence_length -target_length)]
        X.append(X_sample)

    return np.array(X), np.array(Y)


def load_LSTM_data(path='./projectwind/data/'):

    # Load data
    X_train = np.load(f'{path}/LSTM_sequence_train_X_samples.npy')
    y_train = np.load(f'{path}/LSTM_sequence_train_y_samples.npy')
    X_val = np.load(f'{path}/LSTM_sequence_val_X_samples.npy')
    y_val = np.load(f'{path}/LSTM_sequence_val_y_samples.npy')
    X_test = np.load(f'{path}/LSTM_sequence_test_X_samples.npy')
    y_test = np.load(f'{path}/LSTM_sequence_test_y_samples.npy')

    return X_train, y_train, X_val, y_val, X_test, y_test



def get_random_sequences(data, day_length, number_of_subsamples, acceptable_level_of_missing_values):
    """
    Given a dict of "WTG : dataframes" `data`, return a list of sequences of number_of_subsamples and of `day_length` index, with % of acceptable_missing_values.
    """

    sequence_length = day_length*6*24
    last_possible_index = data[0].shape[0] - sequence_length
    counter = 0    # Counter is used as a backstop in case cannot find sequences with acceptable_level_of_missing_values

    sequence_list = []
    for WTG_data in data:
        while len(sequence_list) < number_of_subsamples :
            if counter < 1000:
                # Randomise start
                random_start = np.random.randint(0, last_possible_index)

                # Randomly select sample
                df_sample = WTG_data[random_start: random_start+int(sequence_length)]

                # Verify number of missing values
                if (df_sample.isna().sum()/len(df_sample))[0] < acceptable_level_of_missing_values :

                    # Verify first & last timesteps do not include any missing values (else cannot interpolate)
                    if (df_sample.iloc[0].isna().sum() == 0) & (df_sample.iloc[-1].isna().sum() == 0):

                        # Append to return list
                        sequence_list.append(df_sample)
                        counter = 0 # Reset counter

                # Fetch another sample
                else:
                    counter += 1

            # If cannot find anymore sequences with acceptable_level_of_missing_values
            else:
                break

    # Select random samples with acceptable missing values levels
    return sequence_list
