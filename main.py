from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from serve.predict import predict_landmark_class
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Hand Landmarks Classification API",)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace "*" with your frontend domain for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class LandmarkInput(BaseModel):
    x1: float; y1: float; z1: float
    x2: float; y2: float; z2: float
    x3: float; y3: float; z3: float
    x4: float; y4: float; z4: float
    x5: float; y5: float; z5: float
    x6: float; y6: float; z6: float
    x7: float; y7: float; z7: float
    x8: float; y8: float; z8: float
    x9: float; y9: float; z9: float
    x10: float; y10: float; z10: float
    x11: float; y11: float; z11: float
    x12: float; y12: float; z12: float
    x13: float; y13: float; z13: float
    x14: float; y14: float; z14: float
    x15: float; y15: float; z15: float
    x16: float; y16: float; z16: float
    x17: float; y17: float; z17: float
    x18: float; y18: float; z18: float
    x19: float; y19: float; z19: float
    x20: float; y20: float; z20: float
    x21: float; y21: float; z21: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hand Landmarks Classification API!"}

@app.post("/predict")
def predict(input: LandmarkInput):
    prediction = predict_landmark_class(input)
    return {"prediction": prediction}

# Optional: Only needed if you want to run from code instead of terminal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)