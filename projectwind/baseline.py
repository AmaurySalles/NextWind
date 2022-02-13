import pandas as pd
from sklearn.metrics import mean_absolute_error

from projectwind.clean import add_timestamps
from projectwind.sampling import get_clean_sequences
from projectwind.pipeline import get_pipeline
from projectwind.data import get_data, split_fit_data, split_test_data, get_samples, get_pipeline

def baseline_model() :

    data, fit_data = get_data()
    historical_power = data[0]['Power']
    historical_power = pd.DataFrame(data=historical_power)
    historical_power['Prediction'] = historical_power['Power'].rolling(window = 3).mean()

    baseline = historical_power.dropna()

    y_true = baseline.Power
    y_pred = baseline.Prediction

    mae = mean_absolute_error(y_true, y_pred)

    std = baseline.Power.std()

    return print('mae =' ,mae) , print('Standard deviation =', std)
