
# ✋ Gesture Control Game

This project implements a gesture-controlled game powered by a machine learning model trained on the HAGRID dataset. Players control the game using **hand gestures** captured in real-time through their **webcam**. The gestures are classified by a LightGBM model, and mapped to game directions — bringing a touchless, intuitive interaction experience to life.

## 🎯 Project Objective

The objective of this project is to build and deploy a **real-time gesture classification system** that integrates with a simple maze game, allowing players to navigate using predefined hand gestures. The system is optimized for fast predictions and deployed in a production environment using **FastAPI**.

---

## 🚀 Deployment

✅ The entire system — including the trained model, preprocessing pipeline, and FastAPI server — is deployed on an **AWS EC2 instance**.

📎 **Production API Endpoint**: http://34.227.25.110:8000

📹 The game runs in the browser, communicating with the EC2-hosted API for gesture predictions.

---

## 🧠 ML Model Details

- **Model**: LightGBM Classifier
- **Input**: 63 features extracted using MediaPipe hand landmarks (x, y normalized, z raw)
- **Label Mapping**:
```python
custom_label_map = {
  "one": "up",
  "two": "right",
  "three2": "left",
  "dislike": "down"
}
```
- **Output**: `up`, `down`, `left`, `right`, or `unknown`
- **Preprocessing**: Normalization using saved transformation pipeline (`normalize.pkl`)

---

## 🌳 Git Branching Strategy

- `master`: Production-ready and deployed code
- `research`: Model training, EDA, and experimentation
- `Production`: FastAPI app, monitoring infrastructure, and Docker configs

---

## 🛠 Tech Stack

- **Python 3.10**
- **LightGBM** – ML model for gesture classification
- **MediaPipe** – Landmark extraction from webcam frames
- **FastAPI** – Model serving with REST API
- **Docker & Docker Compose** – Containerized deployment
- **Prometheus & Grafana** – Monitoring and observability
- **AWS EC2** – Production deployment
- **HTML/JS** – Maze game logic and UI

---

## 🎮 Controls

You can control the maze game through:

- 🖐️ Hand gestures (via webcam)
- ⌨️ Keyboard arrow keys (fallback)

---

## 📦 API Endpoints

### `POST /predict`

- **Input**: Image frame from webcam (multipart form)
- **Output**: JSON with predicted direction (`up`, `down`, `left`, `right`, `unknown`)

### `GET /metrics`

- Prometheus metrics like:
  - `prediction_requests_total`
  - `prediction_latency_seconds`

---

## 🧪 ML Model Workflow
![MLflow](https://github.com/user-attachments/assets/dfb55842-9524-4302-98b4-777eedfdf2b5)


### 1. Preprocessing

- MediaPipe extracts 21 landmarks (3D = 63 values)
- Normalized (x, y), z remains raw

### 2. Training

- Model: `LightGBMClassifier`
- Model and encoder saved as `.pkl` files

---



## 🐳 Run Locally with Docker Compose

```bash
git clone https://github.com/your-username/gesture-control-game.git
cd gesture-control-game
docker-compose up --build
```

- **API Docs**: http://localhost:8000/docs  
- **Prometheus**: http://localhost:9090  
- **Grafana**: http://localhost:3000  

---

## 🚀 Quick Game Launch

1. Open `index.html` in your browser (or use Live Server in VS Code).
2. Allow webcam access.
3. Perform gestures to control the maze game.

---

## 📈 Monitoring (Prometheus & Grafana)
![Graffana Dashboard](https://github.com/user-attachments/assets/71aaca19-e1bf-4d3f-860b-819b177a3102)

- Track:
  - API usage
  - Inference latency
  - Request volume
- Easily customizable dashboards available in Grafana




---

## 🤝 Acknowledgments

- HAGRID dataset creators
- MediaPipe by Google
- FastAPI, Docker, Prometheus, and Grafana communities

---

