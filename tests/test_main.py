from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app


client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Hand Landmarks Classification API!"}

def test_predict_endpoint():
    # Sample input based on your LandmarkInput schema
    sample_input = {
  "x1": 109.3924255, "y1": 293.4324667, "z1": 7.01E-08,
  "x2": 117.7148781, "y2": 287.2540718, "z2": -0.013189672,
  "x3": 120.6989937, "y3": 276.2346972, "z3": -0.019938817,
  "x4": 117.0214462, "y4": 267.9602669, "z4": -0.026904851,
  "x5": 108.967289, "y5": 264.9085724, "z5": -0.034252599,
  "x6": 114.5520058, "y6": 258.7795901, "z6": -0.011024829,
  "x7": 117.2078018, "y7": 244.5978928, "z7": -0.023592221,
  "x8": 118.0864449, "y8": 235.101476, "z8": -0.031748686,
  "x9": 117.8812981, "y9": 227.7241496, "z9": -0.037226669,
  "x10": 106.4407082, "y10": 262.3128753, "z10": -0.012771236,
  "x11": 102.6385117, "y11": 255.6150352, "z11": -0.034225296,
  "x12": 106.726078, "y12": 264.4603109, "z12": -0.0414383,
  "x13": 110.6028442, "y13": 271.4732489, "z13": -0.040860083,
  "x14": 99.24468613, "y14": 268.3819569, "z14": -0.015749,
  "x15": 97.86617661, "y15": 265.6281758, "z15": -0.03896011,
  "x16": 104.3458786, "y16": 273.9328469, "z16": -0.037978124,
  "x17": 108.3499146, "y17": 279.4534349, "z17": -0.030664392,
  "x18": 93.23733902, "y18": 276.0187306, "z18": -0.019396113,
  "x19": 92.82515144, "y19": 272.4611671, "z19": -0.036383245,
  "x20": 99.09132385, "y20": 278.0286605, "z20": -0.03340942,
  "x21": 103.3127289, "y21": 282.2862886, "z21": -0.026377864
}

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
