
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="TP3 Big Data - Heart Disease")
objet = joblib.load("model.joblib")

@app.post("/predict")
def predict(patient: dict):
    colonnes = objet["features"]
    manquantes = [c for c in colonnes if c not in patient]

    if manquantes:
        raise HTTPException(
            status_code=400,
            detail={"colonnes_manquantes": manquantes}
        )

    X = pd.DataFrame([patient])[colonnes]
    proba = float(objet["pipeline"].predict_proba(X)[0, 1])

    return {
        "prediction": int(proba >= objet["seuil"]),
        "probabilite": round(proba, 4)
    }
