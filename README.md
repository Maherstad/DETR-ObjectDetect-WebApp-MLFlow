# DETR Object Detection Web Application

<div align="center">
<img src="https://github.com/Maherstad/DETR-ObjectDetect-WebApp-MLFlow/blob/main/assets/od_test.gif" alt="Object Detection Demo" width="800">
</div>

## Overview

Object detection system using DETR (DEtection TRansformer) with MLflow experiment tracking, automated retraining, hyperparameter optimization, and Streamlit web interface. Containerized with Docker for easy deployment.

## Architecture

```
Streamlit Web UI
       ↓
Model Server (MLflow) → DETR Model
       ↓
MLflow Tracking Server
       ↓
SQLite DB + Artifact Storage
```

Three containerized services:
- MLflow Server (port 5050) - experiment tracking
- Model Server (port 7100) - inference API
- Streamlit App (port 8501) - web interface

## Installation

Requirements: Python 3.10+, Docker (optional)

### Local Setup

```bash
# Clone repository
git clone https://github.com/Maherstad/DETR-ObjectDetect-WebApp-MLFlow.git
cd DETR-ObjectDetect-WebApp-MLFlow

# Setup environment
conda create -n detr-mlflow python=3.10
conda activate detr-mlflow
pip install -r requirements.txt
```

**Start services (requires 3 terminals):**

```bash
# Terminal 1: Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5050

# Terminal 2: Register and serve model
# Step 1: Register model first
python object_detection_model_pipeline.py

# Step 2: Serve the model
MLFLOW_TRACKING_URI=http://127.0.0.1:5050 mlflow models serve -m models:/DETR_Object_Detection_Model/Production -p 7100 --no-conda

# Terminal 3: Launch web app
streamlit run webpage.py --server.port 8501
```

**Note**: You must run `object_detection_model_pipeline.py` before serving the model, otherwise you'll get a "Model not found" error.

**Access services:**
- Web App: `http://localhost:8501`
- MLflow UI: `http://localhost:5050`
- Model API: `http://localhost:7100`

### Docker Deployment

```bash
docker-compose up -d
```

**Access services:**
- Web App: `http://localhost:8501`
- MLflow UI: `http://localhost:5050`
- Model API: `http://localhost:7100`

**Manage containers:**
```bash
docker-compose logs -f     # View logs
docker-compose down        # Stop services
```

## Performance

Baseline DETR ResNet-50 on COCO:
- mAP: 0.42 (pre-trained baseline)
- Achievable: 0.89 with fine-tuning and optimization
- Inference: ~150ms/image (CPU), ~30ms (GPU)
- Model size: 167MB

## Advanced Features

### Hyperparameter Optimization

```bash
python hyperparameter_optimization.py
```

Optimizes: learning rate, batch size, epochs, weight decay, optimizer, scheduler, dropout. Results tracked in MLflow.

### Model Evaluation

```bash
python evaluate_model.py
```

Calculates COCO metrics and logs to MLflow.

### Automated Retraining

```bash
python automated_retraining.py
```

Monitors data directory, triggers retraining when thresholds met, auto-promotes better models.

## Project Structure

```
├── object_detection_model_pipeline.py   # Model pipeline & MLflow integration
├── webpage.py                           # Streamlit web app
├── utils.py                             # Helper functions
├── send_request_to_server.py           # API client example
├── evaluate_model.py                    # COCO evaluation
├── hyperparameter_optimization.py       # Optuna tuning
├── automated_retraining.py              # Automated pipeline
├── Dockerfile                           # Container definition
├── docker-compose.yml                   # Multi-container setup
├── requirements.txt                     # Dependencies
└── assets/                              # Sample images
```

## API Usage

```python
import requests
import base64
import pandas as pd

with open("image.jpg", "rb") as f:
    image_bytes = f.read()

data = {"image": [base64.b64encode(image_bytes).decode()]}
csv_data = pd.DataFrame(data).to_csv(index=False)

response = requests.post(
    url="http://localhost:7100/invocations",
    data=csv_data,
    headers={"Content-Type": "text/csv"}
)

result = response.json()
```

## MLflow Dashboard

Access at `http://localhost:5050` for experiment tracking, model versioning, and metrics visualization.

## Tech Stack

- PyTorch 2.1.2 + Transformers 4.37.1
- MLflow 2.10
- Streamlit 1.30
- Optuna 3.5.0
- Docker + Docker Compose
