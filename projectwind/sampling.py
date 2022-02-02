import pandas as pd
import numpy as np

def get_sequences(WTG_data, day_length, number_of_subsamples, acceptable_level_of_missing_values):
    """
    Given a dataframes `data`, return a list of sequences of number_of_subsamples and of `day_length` index, with % of acceptable_missing_values.
    """
    WTG_samples = []


    sequence_length = day_length*6*24
    last_possible_index = WTG_data.shape[0] - sequence_length
    counter = 0    # Counter is used as a backstop in case cannot find sequences with acceptable_level_of_missing_values

    # Select random samples with acceptable missing values levels
    while len(WTG_samples) < number_of_subsamples :
        if counter < 10_000:
            # Randomise start
            random_start = np.random.randint(0, last_possible_index)

            # Randomly select sample
            df_sample = WTG_data[random_start: random_start+int(sequence_length)]

            # Verify number of missing values
            if (df_sample.isna().sum()/len(df_sample))[0] < acceptable_level_of_missing_values :

                # Append to return list
                WTG_samples.append(df_sample)
                counter = 0 # Reset counter

            # Fetch another sample
            else:
                counter += 1

        # If cannot find anymore sequences with acceptable_level_of_missing_values
        else:
            break

    return WTG_samples


def get_clean_sequences(data, fitted_pipeline, day_length, number_of_subsamples, acceptable_level_of_missing_values):
    '''
    Create one single random (X,y) array pair containing clean & scaled data sequences.
    Sequences of number_of_subsamples and of `day_length` index, with % of acceptable_missing_values.
    '''
    sequence_length = int(day_length*6*24)

    sample_list = []
    for WTG_data in data:
        # Collect list of sequences from each WTG - returns scaled & clean data
        WTG_sample_list = get_sequences(WTG_data, day_length, number_of_subsamples/25, acceptable_level_of_missing_values)
        sample_list.append(WTG_sample_list)
    print(WTG_sample_list[0])
    print(sample_list[0])
    # Split into
    Y, X = [], []
    for WTG_sample in sample_list:
        for sample in WTG_sample:
            # Get target
            y_sample = sample['Power'].iloc[sample.shape[0]-72:]
            Y.append(np.array(y_sample))

            # Remove target & scale data
            X_sample = sample[0:(sequence_length -72)]
            print(X_sample)
            print(X_sample.shape)
            scaled_X_sample = fitted_pipeline.transform(X_sample)
            X.append(scaled_X_sample)

    return np.array(X), np.array(Y)
