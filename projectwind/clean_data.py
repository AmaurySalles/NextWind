from lib2to3.pytree import Base
import pandas as pd
import os
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator

def degree(df):
    return FunctionTransformer(lambda x: x if x <=180 else 360 - x )

class Imputer(TransformerMixin, BaseEstimator):
    def __init__(self, list_column):
        self.list_column=list_column

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df=X[self.list_column].interpolate()
        return df




if __name__ == "__main__":
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    rawdir = os.path.join(parentdir,"raw_data")
    df=pd.read_csv(os.path.join(rawdir,"A01.csv"))
    print("ok")
