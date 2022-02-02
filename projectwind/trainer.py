import pandas as pd
import numpy as np

# To store & save training samples
from numpy import save
from numpy import load

from projectwind.clean import add_timestamps
from projectwind.pipeline import get_pipeline

def trainer(sample_size=10_000):
    # Load samples
    X_samples = np.load(f'./projectwind/data/{sample_size}_sequence_X_samples.npy')
    y_samples = np.load(f'./projectwind/data/{sample_size}_sequence_y_samples.npy')

    print(X_samples.shape)
    print(y_samples.shape)


    # Get model

    # Train model

    # Predict (test pipeline)

  

if __name__ == "__main__":
    trainer()
    
