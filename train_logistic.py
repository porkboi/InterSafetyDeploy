import os
import pandas as pd
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch latest prompt data
def fetch_data():
    response = supabase.table("prompts").select("*").execute()
    df = pd.DataFrame(response.data)
    return df[['prompt', 'label']]

def train_and_save():
    df = fetch_data()
    X = df['prompt']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

if __name__ == "__main__":
    train_and_save()
