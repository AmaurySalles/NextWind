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

def degree_converter():
    return FunctionTransformer(lambda x: x if x <=180 else 360 - x )

class Interpolate_Imputer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df=X[self.list_column].interpolate()
        return df

# Pipeline
def get_pipeline():
    
    # Energy
    energy_pipe = Pipeline([
        ('converter', energy_converter()),
        ('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())
    ])
    
    # Wind Speed
    wind_speed_pipe = Pipeline([
        ('imputer', Interpolate_Imputer()),
        ('scaler', StandardScaler())
    ])


    # Nacelle Direction
    nacelle_dir_pipe = Pipeline([
        ('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Misalignment
    misalignment_pipe = Pipeline([
        ('converter', degree_converter()),
        ('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Rotor Speed
    rotor_speed_pipe = Pipeline([
        ('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])

    # Blade Pitch
    blade_pitch_pipe = Pipeline([
        ('imputer', Interpolate_Imputer()),
        ('scaler', MinMaxScaler())

    ])


    # Pre-processor
    preprocessor = ColumnTransformer([
        ('energy', energy_pipe, ['Power']),
        ('wind_speed', wind_speed_pipe, ['Wind Speed']),
        ('nacelle_dir', nacelle_dir_pipe, ['Nacelle Orientation']),
        ('misalignment', misalignment_pipe, ['Misalignment']),
        ('rotor_speed', rotor_speed_pipe, ['Rotor Speed']),
        ('blade_pitch', blade_pitch_pipe, ['Blade Pitch']),
    ])

    return preprocessor