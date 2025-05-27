import joblib
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer import LandmarkNormalizer, map_labels

# Load model
model = joblib.load("model.pkl")

# Create transformer instance
normalizer = LandmarkNormalizer()

# Dummy fit for transformer since it's required (fit on dummy structure)
dummy = pd.DataFrame(columns=[
    f"{axis}{i}" for i in range(1, 22) for axis in ['x', 'y', 'z']
])
dummy.fillna(0, inplace=True)
normalizer.fit(dummy)

def preprocess_input(df):
    df = df.drop(columns=[col for col in df.columns if col.startswith('z')], errors='ignore')
    df = normalizer.transform(df)
    return df
