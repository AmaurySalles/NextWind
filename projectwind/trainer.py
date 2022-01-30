import pandas as pd
import numpy as np
import sklearn

from projectwind.data import get_data, split_test_data, split_fit_data
from projectwind.sampling import split_subsample_sequence
from projectwind.pipeline import get_pipeline

def trainer():

    # Get data & perform splits
    data, fit_data, = get_data()
    train_data, test_data = split_test_data()
    X_fit, y_fit = split_fit_data(fit_data)
    #clean_data = clean_data(train_data)
    
    # Pipeline fit
    pipeline = get_pipeline()
    pipeline.fit(fit_data)
    
    # Sample transform
    scaled_samples = []
    for WTG in train_data:
        X_sample, y_sample = split_subsample_sequence(train_data[WTG],
                                    day_length=5.5, 
                                    numer_of_subsamples=10,
                                    acceptable_level_of_missing_values=0.1)
        
        for i in range(len(X_sample)):
            scaled_samples.append(pipeline.transform(X_sample[i]), y_sample[i])

    print(scaled_samples)
    
    # Save samples for quicker upload time
    samples_csv_zip = pd.DataFrame()
    for sample in scaled_samples:
        samples_csv_zip = pd.concat((samples_csv_zip,sample),ignore_index=False)
    samples_csv_zip.to_csv('./data/samples.csv')

    # Load samples
    
    # Split train/eval samples

    # Get model

    # Train model

    # Predict (test pipeline)

  

if __name__ == "__main__":
    trainer()
    
