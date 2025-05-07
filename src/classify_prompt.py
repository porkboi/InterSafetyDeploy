import numpy as np
from src.utils import load_model, get_supabase_client

def classify(prompt: str, top_k: int = 5):
    model = load_model()
    supabase = get_supabase_client()

    embedding = model.encode(prompt).tolist()

    query = f"""
        SELECT prompt, label, embedding,
        (1 - (embedding <#> '[{','.join(map(str, embedding))}]')) AS similarity
        FROM prompt_vectors
        ORDER BY similarity DESC
        LIMIT {top_k};
    """
    response = supabase.rpc("sql", {"query": query}).execute()
    rows = response.data
    labels = [row["label"] for row in rows]
    return max(set(labels), key=labels.count)
