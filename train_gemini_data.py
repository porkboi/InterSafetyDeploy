import os
import pandas as pd
import json
from supabase import create_client, Client

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def export_to_gemini_format():
    # Fetch data
    response = supabase.table("prompts").select("*").execute()
    df = pd.DataFrame(response.data)

    # Create binary label
    df['label'] = df['positive-ratings'] > df['negative-ratings']

    # Format for fine-tuning: prompt -> classification label
    fine_tune_data = [
        {
            "input": row["prompt"],
            "output": "SAFE" if row["label"] else "DANGEROUS"
        }
        for _, row in df.iterrows()
    ]

    # Write to JSONL
    with open("gemini_finetune.jsonl", "w") as f:
        for item in fine_tune_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Saved {len(fine_tune_data)} examples to gemini_finetune.jsonl")

if __name__ == "__main__":
    export_to_gemini_format()
