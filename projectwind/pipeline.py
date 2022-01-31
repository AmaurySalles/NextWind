import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer

# Customer Transformers
def degree_converter():
    return FunctionTransformer(lambda x: x if x <=180 else 360 - x )

class Interpolate_Imputer(TransformerMixin, BaseEstimator):
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = X[self.list_column].interpolate()
        return X_transformed

# Pipeline
def get_pipeline():
    
    # Wind Speed
    wind_speed_pipe = Pipeline([
        #('imputer', Interpolate_Imputer()),
        ('scaler', StandardScaler())
    ])


    # Nacelle Direction
    nacelle_dir_pipe = Pipeline([
        #('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Misalignment
    misalignment_pipe = Pipeline([
        #('converter', degree_converter()),
        #('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Rotor Speed
    rotor_speed_pipe = Pipeline([
        #('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Blade Pitch
    blade_pitch_pipe = Pipeline([
        #('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])


    # Pre-processor
    preprocessor = ColumnTransformer([
        ('wind_speed', wind_speed_pipe, ['Wind Speed']),
        ('nacelle_dir', nacelle_dir_pipe, ['Nacelle Orientation']),
        ('misalignment', misalignment_pipe, ['Misalignment']),
        ('rotor_speed', rotor_speed_pipe, ['Rotor Speed']),
        ('blade_pitch', blade_pitch_pipe, ['Blade Pitch']),
    ])

    return preprocessor