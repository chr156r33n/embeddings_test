import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import os
from sklearn.metrics import precision_recall_fscore_support

# Load SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_sbert(text):
    return sbert_model.encode(text, normalize_embeddings=True)

def get_embedding_openai(text, api_key):
    try:
        openai.api_key = api_key
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        # Note: Ada embeddings are not normalized
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

def compute_cosine(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_dot(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    return np.dot(vec1, vec2)

st.title("Keyword Similarity Comparison (Cosine vs. Dot Product)")
st.markdown("By Chris Green [www.chris-green.net](https://www.chris-green.net/)")

# Inputs
api_key = st.text_input("OpenAI API Key", type="password")
ground_truth_file = st.file_uploader("Upload Ground Truth CSV (Keyword1, Keyword2, Match 1/0)", type=["csv"])

use_sbert   = st.checkbox("Use SBERT", True)
use_openai = st.checkbox("Use OpenAI", True) and bool(api_key)

if ground_truth_file and st.button("Compute Similarity"):
    ground_truth = pd.read_csv(ground_truth_file)
    true_labels = ground_truth["Match"].tolist()

    # Prepare storage
    results = []
    scores = {
        "sbert_cosine": [],
        "sbert_dot": [],
        "openai_cosine": [],
        "openai_dot": []
    }

    for _, row in ground_truth.iterrows():
        k1, k2, match = row["Keyword1"], row["Keyword2"], row["Match"]
        out = {"Keyword 1": k1, "Keyword 2": k2, "Ground Truth Match": match}

        # SBERT
        if use_sbert:
            e1 = get_embedding_sbert(k1)
            e2 = get_embedding_sbert(k2)
            cos = compute_cosine(e1, e2)
            dot = compute_dot(e1, e2)
            out["SBERT Cosine"] = cos
            out["SBERT Dot"]    = dot
            scores["sbert_cosine"].append(cos)
            scores["sbert_dot"].append(dot)

        # OpenAI
        if use_openai:
            e1 = get_embedding_openai(k1, api_key)
            e2 = get_embedding_openai(k2, api_key)
            cos = compute_cosine(e1, e2)
            dot = compute_dot(e1, e2)
            out["OpenAI Cosine"] = cos
            out["OpenAI Dot"]    = dot
            scores["openai_cosine"].append(cos if cos is not None else 0)
            scores["openai_dot"].append(dot if dot is not None else 0)

        results.append(out)

    # Find best thresholds
    best_thresholds = {}
    for metric, vals in scores.items():
        # filter None
        arr = np.array([v for v in vals if v is not None])
        if arr.size == 0:
            continue
        lb = np.percentile(arr, 10)
        ub = np.percentile(arr, 90)
        best_f1 = 0
        best_t  = lb
        for t in np.linspace(lb, ub, 50):
            preds = [1 if v >= t else 0 for v in vals]
            p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        best_thresholds[metric] = best_t

    # Build DataFrame
    df = pd.DataFrame(results)

    # Classification Errors for each metric
    for model in ["SBERT", "OpenAI"]:
        for mtype in ["Cosine", "Dot"]:
            col_score = f"{model} {mtype}"
            thr = best_thresholds.get(f"{model.lower()}_{mtype.lower()}", 0)
            err_col = f"{model} {mtype} Error"
            if col_score in df.columns:
                df[err_col] = df.apply(
                    lambda r: 
                        "False Positive" if r[col_score] is not None and r[col_score] >= thr and r["Ground Truth Match"] == 0
                        else "False Negative" if r[col_score] is not None and r[col_score] <  thr and r["Ground Truth Match"] == 1
                        else "Correct",
                    axis=1
                )

    # Show table
    st.dataframe(df)

    # Display thresholds
    st.write("### Recommended Thresholds")
    for k,v in best_thresholds.items():
        st.write(f"- **{k.replace('_',' ').title()}**: {v:.4f}")

    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "similarity_results.csv", "text/csv")
