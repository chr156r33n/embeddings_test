import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import os

# Load models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_sbert(text):
    return sbert_model.encode(text, normalize_embeddings=True)

def get_embedding_openai(text, api_key):
    openai.api_key = api_key
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

def compute_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

st.title("Keyword Similarity Comparison")

# User Inputs
api_key = st.text_input("OpenAI API Key", type="password")
keyword_list_1 = st.text_area("Keyword List 1 (one per line)").split("\n")
keyword_list_2 = st.text_area("Keyword List 2 (one per line)").split("\n")

# Model Selection
use_sbert = st.checkbox("Use SBERT", True)
use_openai = st.checkbox("Use OpenAI", True)

if st.button("Compute Similarity"):
    results = []
    for k1 in keyword_list_1:
        for k2 in keyword_list_2:
            row = [k1, k2]
            sbert_score, openai_score, percent_diff = None, None, None
            if use_sbert:
                sbert_score = compute_similarity(get_embedding_sbert(k1), get_embedding_sbert(k2))
                row.append(sbert_score)
            if use_openai and api_key:
                openai_score = compute_similarity(get_embedding_openai(k1, api_key), get_embedding_openai(k2, api_key))
                row.append(openai_score)
            if sbert_score is not None and openai_score is not None:
                percent_diff = abs(sbert_score - openai_score) / ((sbert_score + openai_score) / 2) * 100
                row.append(percent_diff)
            results.append(row)
    
    # Dataframe Output
    columns = ["Keyword 1", "Keyword 2"]
    if use_sbert:
        columns.append("SBERT Similarity")
    if use_openai and api_key:
        columns.append("OpenAI Similarity")
    if use_sbert and use_openai:
        columns.append("% Difference")
    
    df = pd.DataFrame(results, columns=columns)
    st.dataframe(df)
    
    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "similarity_results.csv", "text/csv")
