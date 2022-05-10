import pandas as pd
import numpy as np

# To store & save training samples
from numpy import save
from numpy import load

# Loss graphs
import matplotlib.pyplot as plt

from projectwind.ss.LSTM_preproc import init_LSTM_data, load_LSTM_data
from projectwind.ss.LSTM_model import init_LSTM_model

from tensorflow.keras.callbacks import EarlyStopping

def LSTM_trainer():

    # Load data       - TODO fetch all at once (X_train, X_val, X_test, y_train, y_val, y_test) for a specific sample size
    X_train, y_train, X_val, y_val, X_test, y_test = load_LSTM_data()
    #X_test, y_test = load_LSTM_test_data()
    
    # Get model
    model = init_LSTM_model(n_steps_in=X_train.shape[1], n_steps_out=y_train.shape[1], n_features=X_train.shape[2])

    # Train model    - TODO: turn this into a function we can fine-tune for every new run
    es = EarlyStopping(patience=2, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val), 
                        #batch_size=1,
                        epochs=10,
                        callbacks=[es],
                        verbose=1)
    plot_loss_mse(history)

    # Predict (test pipeline)  - TODO: X_test pipeline
    #y_pred = model.predict(X_test)

    return history

def plot_loss(history):
    fig , ax = plt.subplots(figsize=(13,4))
    
    # Loss plot 
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylim(ymin=0)
    ax.legend(['Train', 'Validation'], loc='best')
    ax.grid(axis="x",linewidth=0.5)
    ax.grid(axis="y",linewidth=0.5)      

    plt.show()

if __name__ == "__main__":
    LTSM_trainer()
    
