import pandas as pd
from src.utils import load_model, get_supabase_client

model = load_model()
supabase = get_supabase_client()

# Fetch prompt data
response = supabase.table("prompts").select("*").execute()
df = pd.DataFrame(response.data)

# Labeling logic
df['label'] = df.apply(
    lambda row: "SAFE" if row["positive-ratings"] > row["negative-ratings"] else "DANGEROUS",
    axis=1
)

# Upload with embeddings
for _, row in df.iterrows():
    embedding = model.encode(row["prompt"]).tolist()
    supabase.table("prompt_vectors").insert({
        "prompt": row["prompt"],
        "label": row["label"],
        "embedding": embedding
    }).execute()
