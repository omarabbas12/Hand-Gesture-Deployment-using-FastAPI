# transformer.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LandmarkNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.x_columns = []
        self.y_columns = []

    def fit(self, X, y=None):
        self.x_columns = [col for col in X.columns if col.startswith('x')]
        self.y_columns = [col for col in X.columns if col.startswith('y')]
        return self

    def transform(self, X):
        X = X.copy()

        # Center relative to (x1, y1)
        X[self.x_columns] = X[self.x_columns].sub(X['x1'], axis=0)
        X[self.y_columns] = X[self.y_columns].sub(X['y1'], axis=0)

        # Distance to point 13
        euclidean_distance = np.sqrt(
            (X['x13'])**2 + (X['y13'])**2
        ).replace(0, 1e-9)

        # Normalize
        X[self.x_columns] = X[self.x_columns].div(euclidean_distance, axis=0)
        X[self.y_columns] = X[self.y_columns].div(euclidean_distance, axis=0)

        return X

def map_labels(df):
    """
    Maps raw labels to a limited set of 5 directional categories.
    Adds a new column 'label' to the dataframe with mapped values.
    """
    label_mapping = {
        'one': 'up',
        'dislike': 'down',
        'three2': 'left',
        'peace': 'right',
        'two_up': 'right',
        'fist': 'stop',
        'stop': 'stop',
        'palm': 'stop'
    }
    df = df.copy()
    df['label'] = df['label'].apply(lambda x: label_mapping.get(x, 'unknown'))
    return df
