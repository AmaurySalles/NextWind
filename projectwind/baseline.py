import pandas as pd
from sklearn.metrics import mean_absolute_error

from projectwind.data import get_WTG_cat_data

def baseline_model(n_steps_out):

    # Fetch all 25xWTG power data
    data = get_WTG_cat_data('Power')
    
    print('### Calculating baseline loss ###')

    # Calculate average of WTG power
    y_true = data.mean(axis=1)

    # Baseline: prediction equates to last 6 hours rolling average
    y_pred = y_true.rolling(window = n_steps_out).mean().shift(n_steps_out)
    y_pred = y_pred.dropna()

    #Align timestep indices
    y_true = y_true.loc[y_pred.index[0]:y_pred.index[-1]]

    mae = mean_absolute_error(y_true, y_pred)
    print('MAE =', mae.round())
    std = y_true.std()
    print('Target std =', std.round())

    return mae, std
