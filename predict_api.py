from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.classify_prompt import classify
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "Prompt Safety API is running"}

@app.post("/classify")
async def classify_prompt_api(request: PromptRequest):
    try:
        result = classify(request.prompt)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
