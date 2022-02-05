import pandas as pd
import numpy as np

# To store & save training samples
from numpy import save
from numpy import load

from projectwind.data import get_samples
from projectwind.pipeline import get_pipeline
from projectwind.LSTM_model import get_LSTM_model

def trainer():

    # Load data
    X_train, y_train = get_samples(model_type='DL', sample_size=10_000)
    print(X_train.shape)
    print(y_train.shape)

    # Get model
    model = get_LSTM_model(n_steps_in=X_train.shape[1], n_steps_out=y_train.shape[1], n_features=X_train.shape[2])

    # Train model

    # Predict (test pipeline)

    return X_train, y_train

if __name__ == "__main__":
    trainer()
    
