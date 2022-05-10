import tensorflow as tf

from nextwind.preproc import make_datasets
from nextwind.preproc import SequenceGenerator
from nextwind.models import Baseline_model

def trainer():

    # Retrieve data & split into train, val & test sets
    datasets = make_datasets(forecast_data='MERRA2') # Tuple containing train, val & test sets

    # Create sequence window
    n_steps_in = 24
    n_steps_out = 6
    window = SequenceGenerator(n_steps_in, n_steps_out, n_steps_out,
                          datasets, classification=True,
                          forecast_columns=['Forecast_wind_speed', 'Forecast_X', 'Forecast_Y'],
                          label_columns=['Power'])

    # Retrieve sequences
    X_train, y_train, X_fc_train = window.train.values()
    X_val, y_val, X_fc_val = window.val.values()
    X_test, y_test, X_fc_test = window.test.values()

    # Init baseline model
    baseline = Baseline_model(window)

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    val_performance = baseline.evaluate(x=window.val['X'], y=window.val['y'], verbose=1)
    test_performance = baseline.evaluate(x=window.test['X'], y=window.test['y'], verbose=1)
    print(val_performance)
    print(test_performance)

    window.plot(baseline)

if __name__ == "__main__":
    trainer()
    
