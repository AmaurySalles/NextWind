import pandas as pd
import numpy as np

# To store & save training samples
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

from projectwind.data import get_data, split_test_data, split_fit_data
from projectwind.clean import add_timestamps
from projectwind.sampling import get_clean_sequences
from projectwind.pipeline import get_pipeline

def trainer(fetch_new_data=False, day_length=5.5, number_of_subsamples=100, acceptable_level_of_missing_values=0.05):

    if fetch_new_data == True:
        # Get data & perform  splits
        data, fit_data, = get_data()
        #data = add_timestamps(data)   # TODO Does not work - need to integrate function, so that it runs dict("WTG":pd.DataFrame)
        train_data, test_data = split_test_data(data)
        X_fit, y_fit = split_fit_data(fit_data)

        
        # Pipeline fit
        pipeline = get_pipeline()
        pipeline.fit(X_fit)
        
        # Transform data & fetch sequences
        samples = get_clean_sequences(train_data,
                                    fitted_pipeline=pipeline,
                                    day_length=day_length, 
                                    number_of_subsamples=number_of_subsamples,  # Starting small with only 100 samples / WTG
                                    acceptable_level_of_missing_values=acceptable_level_of_missing_values)

        
        print(samples) # 3D array with sequences, timesteps & features
    
        # Shuffle WTG sequences & target

        
        # Save sequences for quicker upload time8
        #data = asarray(samples)
        savetxt(f'./data/sequence_samples.csv', data, delimiter=',')
    
        # samples_csv_zip = pd.DataFrame()
        # for sample in scaled_samples:
        #     samples_csv_zip = pd.concat((samples_csv_zip,sample),ignore_index=False)
        # samples_csv_zip.to_csv('./data/samples.csv')

    
    # Load samples
    else:
        data = loadtxt(f'./data/sequence_samples.csv', delimiter=',')


    # Get model

    # Train model

    # Predict (test pipeline)

  

if __name__ == "__main__":
    trainer()
    
