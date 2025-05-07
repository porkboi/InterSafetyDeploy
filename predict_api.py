from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

class PromptInput(BaseModel):
    prompt: str

@app.post("/predict")
def predict(input: PromptInput):
    try:
        X = vectorizer.transform([input.prompt])
        pred = model.predict(X)[0]
        return {"prediction": "dangerous" if pred == 1 else "not_dangerous"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
