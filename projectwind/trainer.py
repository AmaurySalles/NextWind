import pandas as pd
import numpy as np
import sklearn

from projectwind.clean import clean_data
from projectwind.clean import pipeline

# Get data & split test data

# Get sequence samples

# Clean sequence samples (remaining missing timesteps)

# Scale sequence samples

# Split train/eval samples

# Get model

# Train model

# Predict (test pipeline)



def get_data():
    data = pd.read_csv("./raw_data/A01.csv")
    #data = clean_data(data)
    return data

def prep_data():
    pass    

if __name__ == "__main__":

    # Pipeline fit
    full_data = get_data()
    pipeline = pipeline()
    pipeline.fit(full_data)
    
    # Sample transform
    samples = fetch_samples()
    scaled_sampled = []
    for data_sample in samples:
        scaled_sampled.append(pipeline.transform(data_sample))

