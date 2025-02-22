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
use_openai = st.checkbox("Use OpenAI", True) and bool(api_key)

if ground_truth_file and st.button("Compute Similarity"):
    ground_truth = pd.read_csv(ground_truth_file)
    results = []
    best_thresholds = {}
    model_scores = {}
    true_labels = ground_truth["Match"].tolist()
    
    if use_sbert:
        model_scores["sbert"] = []
    if use_openai:
        model_scores["openai"] = []
    
    for _, row in ground_truth.iterrows():
        k1, k2, match = row["Keyword1"], row["Keyword2"], row["Match"]
        result_row = [k1, k2, match]
        
        if use_sbert:
            sbert_score = compute_similarity(get_embedding_sbert(k1), get_embedding_sbert(k2))
            result_row.append(sbert_score)
            model_scores["sbert"].append(sbert_score)
        
        if use_openai:
            openai_score = compute_similarity(get_embedding_openai(k1, api_key), get_embedding_openai(k2, api_key))
            result_row.append(openai_score)
            model_scores["openai"].append(openai_score)
        
        results.append(result_row)
    
    # Find best threshold for each model using percentile-based thresholding
    for model in model_scores:
        score_array = np.array(model_scores[model])
        lower_bound = np.percentile(score_array, 10)  # 10th percentile as lower bound
        upper_bound = np.percentile(score_array, 90)  # 90th percentile as upper bound
        best_f1 = 0
        best_threshold = lower_bound
        for threshold in np.linspace(lower_bound, upper_bound, 50):  # 50 steps within range
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
        columns.append("SBERT Classification Error")
    if use_openai:
        columns.append("OpenAI Similarity")
        columns.append("OpenAI Classification Error")
    
    df = pd.DataFrame(results, columns=columns[:len(results[0])])  # Dynamically adjust columns
    
    # Compute classification errors
    if use_sbert:
        df["SBERT Classification Error"] = df.apply(lambda row: "False Positive" if row["SBERT Similarity"] >= best_thresholds["sbert"] and row["Ground Truth Match"] == 0 else "False Negative" if row["SBERT Similarity"] < best_thresholds["sbert"] and row["Ground Truth Match"] == 1 else "Correct", axis=1)
    
    if use_openai and "OpenAI Similarity" in df.columns:
        df["OpenAI Classification Error"] = df.apply(lambda row: "False Positive" if row["OpenAI Similarity"] >= best_thresholds["openai"] and row["Ground Truth Match"] == 0 else "False Negative" if row["OpenAI Similarity"] < best_thresholds["openai"] and row["Ground Truth Match"] == 1 else "Correct", axis=1)
    
    st.dataframe(df)
    
    # Display best thresholds
    if best_thresholds:
        st.write("### Recommended Thresholds (Percentile-Based)")
        for model, threshold in best_thresholds.items():
            st.write(f"{model.upper()} Best Threshold: {threshold:.2f}")
    
    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "similarity_results.csv", "text/csv")
