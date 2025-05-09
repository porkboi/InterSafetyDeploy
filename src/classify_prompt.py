import numpy as np
from src.utils import load_model, get_supabase_client

def classify(prompt: str, top_k: int = 5):
    try:
        print("📥 Classifying prompt:", prompt)
        model = load_model()
        supabase = get_supabase_client()
        embedding = model.encode(prompt).tolist()
        print("🔢 Embedding generated:", embedding[:5])

        response = supabase.rpc("classify_prompt_vec", {
            "embedding": embedding,
            "top_k": top_k
        }).execute()

        if response.error:
            raise Exception("Supabase RPC error")

        label = response.data[0]["label"]
        return label

    except Exception as e:
        print("❌ classify() error:", str(e))
        raise

