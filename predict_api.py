from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.classify_prompt import classify

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "Prompt Safety API is running"}

@app.post("/classify")
def classify_prompt_api(request: PromptRequest):
    try:
        result = classify(request.prompt)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
