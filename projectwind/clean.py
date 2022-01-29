import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer

def add_timestamps():
    results = {}
    for file in dataframes.keys():
        df = dataframes[file]
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df.set_index('Fecha',inplace=True)
        ref_date_range = pd.date_range(start='1/05/2019', end='30/09/2021',freq='10T')
        ref_df = pd.DataFrame(index=ref_date_range)
        clean_data = df.reindex(ref_df.index)
        new_df = pd.merge(ref_df,clean_data,left_index=True, right_index=True,how='outer')
        results[file] = new_df
    results['A02.csv'].drop(columns='Unnamed: 7',inplace= True)
    return results

# def fill_na_with_mean(column):
#     column.fillna('', inplace=True)
#     for i in range(len(column)):
#           if column[i] == '' :
#                 column[i] = (column[i-1] + column[i+1])/2
#     return column


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

    # Energy
    energy_pipe = Pipeline([
        ('converter', energy_converter()),
        ('imputer', Imputer()),
        ('scaler', MinMaxScaler())
    ])
    
    # Wind Speed
    wind_speed_pipe = Pipeline([
        ('imputer', Imputer()),
        ('scaler', StandardScaler())
    ])

    # Nacelle Direction
    nacelle_dir_pipe = Pipeline([
        ('converter', degree_converter()),
        ('imputer', Imputer()),
        ('scaler', MinMaxScaler())

    ])

    

    # Pre-processor
    preprocessor = ColumnTransformer([
        ('energy', energy_pipe, ['Power']),
        ('wind speed', wind_speed_pipe, ['Wind Speed']),
        ('Nacelle dir', nacelle_dir_pipe, ['Nacelle Direction'])
    ])

    return preprocessor


    
    