import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from nextwind.preproc import make_datasets
from nextwind.preproc import SequenceGenerator
from nextwind.models import Baseline_model, lstm_regressor_model

def trainer():

    # Retrieve data & split into train, val & test sets
    datasets = make_datasets(forecast_data='MERRA2') # Tuple containing train, val & test sets

    # Create sequence window
    n_steps_in = 24
    n_steps_out = 6
    window = SequenceGenerator(n_steps_in, n_steps_out, n_steps_out,
                          datasets,
                          forecast_columns=['Forecast_wind_speed', 'Forecast_X', 'Forecast_Y'],
                          label_columns=['Power'])

    # Retrieve sequences
    train, val, test = window.get_sequences()

    # Init model scores
    val_performance = {}
    test_performance = {}

    # Init baseline model
    baseline = Baseline_model(window)

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    val_performance['baseline'] = baseline.evaluate(x=val['X'], y=val['y'], verbose=1)
    test_performance['baseline'] = baseline.evaluate(x=test['X'], y=test['y'], verbose=1)

    plot_examples(baseline, window)

    # Init LSTM model
    lstm_model = lstm_regressor_model(window)

    history = compile_and_fit(lstm_model, window, epoch=5)
    val_performance['lstm_model'] = lstm_model.evaluate(x=val['X'], y=val['y'], verbose=1)
    test_performance['lstm_model'] = lstm_model.evaluate(x=test['X'], y=test['y'], verbose=1)

    plot_loss(history)
    plot_examples(lstm_model, window)

    


def compile_and_fit(model, window, patience=5, epoch=20, name=None, forecast=True):
    """
    Compiles given model, and fits with given window's data.
    """
    # Fetch inputs
    train_inputs = [window.train['X']]
    val_inputs = [window.val['X']]

    if forecast:
        train_inputs.append(window.train['X_fc'])
        val_inputs.append(window.val['X_fc'])

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    restore_best_weights=True)
    callbacks = [early_stopping]

    # Reduce learning rate by an order of magnitude if val_loss does not improve for 20 epoch
    # rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
    #                                             factor=0.1,
    #                                             min_lr=1e-7,
    #                                             verbose=1,
    #                                             patience=10)
    # callbacks.append(rlrop)

    # Model checkpoint
    if name:
        checkpoint=tf.keras.callbacks.ModelCheckpoint(f"./checkpoint/Feedback_Model_{name}.h5", 
                                                    save_best_only=True,
                                                    save_weights_only=True)
        callbacks.append(checkpoint)

    # Compile
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    # Fit
    history = model.fit(x = train_inputs, y=window.train['y'],
                        validation_data= [val_inputs, window.val['y']],
                        epochs=epoch, callbacks=callbacks)
    
    return history


def plot_examples(model, window, n=3, forecast=True):
    """
    Graphically represent up to 10 examples of the historical target (part of inputs), target and predictions from the validation dataset 
    Feed a specific model to get its predictions, with specified input data & data, else it will fetch example (random) val set data
    ---------
    Parameters:
    model: 'Object'
            Class object of the trained model to use & predict from
    window: 'Class Object'
            SequenceGenerator class object, containing size of window & all datasets' sequences
    n: 'Int'
        Number of examples to display. Must be an int between 0 and 10.
    ---------
    Returns:
    A graph showing inputs, target and prediction of the specified model 
    """
    # Graphical inputs use 3 random validation sets
    i = np.random.randint(0,len(window.example['X']), size=n)
    inputs = window.example['X'][i]
    label = window.example['y'][i]

    # Figure out label name & index (first target, if there are a multiple)
    target_col = window.label_columns[0]
    target_icol = window.column_indices[target_col]
    
    # Find predictions
    try:
        if (window.forecast_columns is None) or (forecast is False):
            predictions = model.predict(inputs)
        else:    
            fc_inputs = window.example['X_fc'][i]
            predictions = model.predict([inputs, fc_inputs])
    except KeyError:
        return print("Please provide forecast inputs in SequenceGenerator (load=False) & retry. Else turn forecast to False") 


    # Plots 
    plt.figure(figsize=(12, 8))
    for n in range(3):
        # Increase subplot number
        plt.subplot(3, 1, n+1)

        # Plot historical target (part of inputs)
        plt.plot(window.input_indices, inputs[n, :, target_icol],
                label='Inputs', marker='.', zorder=-10)

        # Plot label / target
        plt.plot(window.label_indices, label[n, :, target_icol],
                    label='Labels', c='#2ca02c', marker='.')

        # Plot predictions
        plt.plot(window.label_indices, predictions[n, :, target_icol],
                    marker='X', label='Predictions', c='#ff7f0e')

        # Legend, axis labels & others
        if n == 0:
            plt.legend()
        
        plt.ylabel(f'{target_col}')
        plt.xlabel('Time [h]')
        plt.tight_layout()


def plot_loss(history):
    """
    Graphically represent training and validation set loss history 
    ---------
    Parameters:
    history: 'Object'
            Class object of the model's training history (output from compile_and_fit function)
    ---------
    Returns:
    A graph showing inputs, target and prediction of the specified model 
    """
    # summarize history for accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model MAE')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()  



if __name__ == "__main__":
    trainer()
    
