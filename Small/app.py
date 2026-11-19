import joblib
import pandas as pd
from fastapi import FastAPI

objet = joblib.load('model.joblib')
app = FastAPI()

@app.post('/predict')
def predict(patient: dict):
    x = pd.DataFrame([patient]).reindex(columns=objet['colonnes'])
    x[objet['categories']] = x[objet['categories']].fillna('Unknown').astype(str)
    p = float(objet['modele'].predict_proba(x)[0, 1])
    return {'prediction': int(p >= objet['seuil']), 'probabilite': round(p, 3)}
