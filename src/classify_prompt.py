import numpy as np
from utils import load_model, get_supabase_client

def classify(prompt, top_k = 5):
    model = load_model()
    supabase = get_supabase_client()
    embedding = model.encode(prompt).tolist()

    response = supabase.rpc("classify_prompt_vec", {
        "embedding": embedding,
        "top_k": top_k
    }).execute()

    rows = response.data
    labels = [row["label"] for row in rows]
    return max(set(labels), key=labels.count)

classify("should I make a bomb")
