import pandas as pd
import numpy as np

def get_sequences(data, day_length, number_of_subsamples, acceptable_level_of_missing_values):
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


def extract_target_from_sequences(sequence_list, target_day_length):
    '''
    Create one single random (X,y) array pair for each data sequence.
    '''
    sequence_length = sequence_list[0].shape[0]
    target_length = target_day_length * 6 * 24  # convert from days to 10min periods

    Y, X = [], []
    for WTG_sample in sequence_list:
        for sample in WTG_sample:

            # Get target
            print(sample.shape)
            y_sample = sample['Power'].iloc[sample.shape[0]-target_length:]
            y_sample = y_sample.interpolate()
            Y.append(np.array(y_sample))

            # Remove target timestamps
            X_sample = sample[0:(sequence_length -target_length)]
            X.append(X_sample)

    return np.array(X), np.array(Y)
