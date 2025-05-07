import os
from supabase import create_client
from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variable")
    return create_client(url, key)
