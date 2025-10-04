# Iris Classifier MLOps Project

## Overview
End-to-end MLOps pipeline demonstrating:
- Data ingestion
- Preprocessing
- Model training with MLflow tracking
- REST API deployment using FastAPI
- Docker deployment

## Run Locally

1. Train the model:
```bash
python src/model/train.py
```
2. Test
```
POST to http://localhost:8000/predict/
 with JSON:
{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
```

