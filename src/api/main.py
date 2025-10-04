from fastapi import FastAPI
import joblib
import pandas as pd


app = FastAPI(title="Iris Classifier API")

model = joblib.load("../model/model.pkl")


@app.post("/predict/")
async def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}