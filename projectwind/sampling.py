import pandas as pd
import numpy as np

########## CHANGES! #############

def get_sequences(WTG_data, day_length, number_of_subsamples, acceptable_level_of_missing_values = 0.1):
    """
    Given a dataframes `data`, return a list of sequences of number_of_subsamples and of `day_length` index, with % of acceptable_missing_values.
    """
    WTG_samples = []

    # Randomise start
    sequence_length = day_length*6*24
    last_possible_index = WTG_data.shape[0] - sequence_length
    random_start = np.random.randint(0, last_possible_index)

    # Select random samples with acceptable missing values levels 
    while len(WTG_samples) < number_of_subsamples :

        # Randomly select sample            
        df_sample = WTG_data[random_start: random_start+int(sequence_length)]

        # Verify number of missing values
        if (df_sample.isna().sum()/len(df_sample))[0] < acceptable_level_of_missing_values :
            
            # Append to return list
            WTG_samples.append(df_sample)

    return WTG_samples


def get_clean_sequences(data, fitted_pipeline, day_length, number_of_subsamples, acceptable_level_of_missing_values = 0.1):
    '''
    Create one single random (X,y) array pair containing clean & scaled data sequences.
    Sequences of number_of_subsamples and of `day_length` index, with % of acceptable_missing_values.
    '''
    sequence_length = int(day_length*6*24)
    
    # Collect list of sequences from each WTG - returns scaled & clean data
    sequence_list = get_sequences(data, day_length, number_of_subsamples, acceptable_level_of_missing_values = 0.1)
    
    # Split into 
    Y, X = [], []
    for sample in sequence_list:

        # Get target 
        y_sample = sample['Power'].iloc[sample.shape[0]-72:]
        Y.append(np.array(y_sample))
        
        # Remove target & scale data
        X_sample = sample[0:(sequence_length -72)]
        scaled_X_sample = fitted_pipeline.transform(X_sample)
        X.append(scaled_X_sample)
    
    return np.array(X), np.array(Y)