import os
from supabase import create_client
from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)
