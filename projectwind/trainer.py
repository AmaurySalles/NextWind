import pandas as pd
import numpy as np

# To store & save training samples
from numpy import save
from numpy import load

from projectwind.data import get_samples
from projectwind.pipeline import get_pipeline

def trainer():

    # Load data
    X_train, y_train = get_samples(model_type='DL', sample_size=10_000)
    print(X_train.shape)
    print(y_train.shape)

    # Get model

    # Train model

    # Predict (test pipeline)

    return X_train, y_train

if __name__ == "__main__":
    trainer()
    
