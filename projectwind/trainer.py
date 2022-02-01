import pandas as pd
import numpy as np

# To store & save training samples
from numpy import asarray
from numpy import save
from numpy import load

from projectwind.data import get_data, split_test_data, split_fit_data
from projectwind.clean import add_timestamps
from projectwind.sampling import get_clean_sequences
from projectwind.pipeline import get_pipeline

def trainer(fetch_new_data=False, day_length=5.5, number_of_subsamples=10_000, acceptable_level_of_missing_values=0.05):
    
    if fetch_new_data == True:
        # Get data & perform  splits
        data, fit_data, = get_data()
        data = add_timestamps(data)   
        train_data, test_data = split_test_data(data)
        X_fit, y_fit = split_fit_data(fit_data)

        
        # Pipeline fit
        pipeline = get_pipeline()
        pipeline.fit(X_fit)
        
        # Transform data & fetch sequences
        samples = get_clean_sequences(train_data,
                                    fitted_pipeline=pipeline,
                                    day_length=day_length, 
                                    number_of_subsamples=number_of_subsamples,  
                                    acceptable_level_of_missing_values=acceptable_level_of_missing_values)

        
        #print(samples.shape) # 3D array with sequences, timesteps & features
    
        # Shuffle WTG sequences & target
        # X_train seed=42
        # y_train seed=42
        
        # Save sequences for quicker upload time
        X, Y = samples
        
        print(X.shape)
        print(Y.shape)

        np.save(f'./projectwind/data/{number_of_subsamples}_sequence_X_samples.npy', X)
        np.save(f'./projectwind/data/{number_of_subsamples}_sequence_y_samples.npy', Y)

    # Load samples
    else:
        X_samples = np.load(f'./projectwind/data/{number_of_subsamples}_sequence_X_samples.npy')
        y_samples = np.load(f'./projectwind/data/{number_of_subsamples}_sequence_y_samples.npy')

        print(X_samples.shape)
        print(y_samples.shape)


    # Get model

    # Train model

    # Predict (test pipeline)

  

if __name__ == "__main__":
    trainer()
    
