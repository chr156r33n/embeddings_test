import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import os
from sklearn.metrics import precision_recall_fscore_support

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
ground_truth_file = st.file_uploader("Upload Ground Truth CSV (Keyword1, Keyword2, Match 1/0)", type=["csv"])

# Model Selection
use_sbert = st.checkbox("Use SBERT", True)
use_openai = st.checkbox("Use OpenAI", True)

if ground_truth_file and st.button("Compute Similarity"):
    ground_truth = pd.read_csv(ground_truth_file)
    results = []
    best_thresholds = {}
    thresholds = np.linspace(0, 1, 101)  # 0.00 to 1.00 in steps of 0.01
    model_scores = {"sbert": [], "openai": []}
    true_labels = ground_truth["Match"].tolist()
    
    for _, row in ground_truth.iterrows():
        k1, k2, match = row["Keyword1"], row["Keyword2"], row["Match"]
        result_row = [k1, k2, match]
        sbert_score, openai_score = None, None
        if use_sbert:
            sbert_score = compute_similarity(get_embedding_sbert(k1), get_embedding_sbert(k2))
            result_row.append(sbert_score)
            model_scores["sbert"].append(sbert_score)
        if use_openai and api_key:
            openai_score = compute_similarity(get_embedding_openai(k1, api_key), get_embedding_openai(k2, api_key))
            result_row.append(openai_score)
            model_scores["openai"].append(openai_score)
        results.append(result_row)
    
    # Find best threshold for each model
    for model in model_scores:
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            predictions = [1 if score >= threshold else 0 for score in model_scores[model]]
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        best_thresholds[model] = best_threshold
    
    # Dataframe Output
    columns = ["Keyword 1", "Keyword 2", "Ground Truth Match"]
    if use_sbert:
        columns.append("SBERT Similarity")
    if use_openai:
        columns.append("OpenAI Similarity")
    
    df = pd.DataFrame(results, columns=columns)
    st.dataframe(df)
    
    # Display best thresholds
    if best_thresholds:
        st.write("### Recommended Thresholds")
        for model, threshold in best_thresholds.items():
            st.write(f"{model.upper()} Best Threshold: {threshold:.2f}")
    
    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "similarity_results.csv", "text/csv")
