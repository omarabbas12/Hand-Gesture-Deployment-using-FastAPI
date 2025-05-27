from utils import model, preprocess_input
from pydantic import BaseModel
import pandas as pd
import joblib

odel = joblib.load("Model.pkl")
transform = joblib.load("normalizer.pkl")

# Define the exact feature names used during training
FEATURE_NAMES = [
    f"{axis}{i}" for i in range(1, 22) for axis in ["x", "y", "z"]
]

def predict_landmark_class(input):
    # Convert input to dict then DataFrame
    data = pd.DataFrame([input.dict()], columns=FEATURE_NAMES)

    # Apply transformation
    processed_df = transform.transform(data)

    # Predict
    prediction = model.predict(processed_df)[0]
    return prediction