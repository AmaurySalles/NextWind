import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.compose import ColumnTransformer


# Customer Transformers
def energy_converter():
    return ColumnTransformer(lambda x: x/6)

def degree_converter(df):
    return FunctionTransformer(lambda x: x if x <=180 else 360 - x )

class Imputer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df=X[self.list_column].interpolate()
        return df

# Pipeline
def pipeline(data):

    energy_pipe = Pipeline([
        ('converter', energy_converter()),
        ('imputer', Imputer()),
        ('scaler', MinMaxScaler())
    ])
    
    wind_speed_pipe = Pipeline([
        ('imputer', Imputer()),
        ('scaler', StandardScaler())
    ])

    nacelle_dir_pipe = Pipeline([
        ('converter', degree_converter()),
        ('imputer', Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Paralellize "num_transformer" and "One hot encoder"
    preprocessor = ColumnTransformer([
        ('energy', energy_pipe, ['Power']),
        ('wind speed', wind_speed_pipe, ['Wind Speed']),
        ('Nacelle dir', nacelle_dir_pipe, ['Nacelle Direction'])
    ])

    return preprocessor

if __name__ == "__main__":


    pipeline.fit(all_data)
    pipeline.transform(for i in sequence)
    
    