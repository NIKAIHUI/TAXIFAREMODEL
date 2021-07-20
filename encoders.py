from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
from scipy.spatial import minkowski_distance

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    
    def __init__(self, time_column, time_zone_name='America/Los_Angeles'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["hour"] = X.index.hour
        return X[[
                'hour',
            ]].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X['distance'] = minkowski_distance(X, p=1)
        return X[[
                'distance',
            ]]

