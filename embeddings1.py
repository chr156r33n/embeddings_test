import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics import precision_recall_fscore_support

# Load SBERT (raw embeddings)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_sbert(text):
    # normalize_embeddings=False → return raw vectors
    return sbert_model.encode(text, normalize_embeddings=False)

def get_embedding_openai(text, api_key):
    try:
        openai.api_key = api_key
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return np.array(resp.data[0].embedding)
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

st.title("Keyword Similarity: Cosine vs. Dot Product")
st.markdown("By Chris Green — [Website](https://www.chris-green.net/)")

# User inputs
api_key = st.text_input("OpenAI API Key", type="password")
gt_file = st.file_uploader("Upload Ground Truth CSV (Keyword1, Keyword2, Match 1/0)", type=["csv"])

use_sbert   = st.checkbox("Use SBERT", True)
use_openai  = st.checkbox("Use OpenAI", True) and bool(api_key)

if gt_file and st.button("Compute Similarity"):
    # Load and normalize columns
    df_gt = pd.read_csv(gt_file)
    df_gt.columns = [c.strip().lower() for c in df_gt.columns]
    required = {"keyword1","keyword2","match"}
    if not required.issubset(set(df_gt.columns)):
        st.error(f"Missing columns: {', '.join(required - set(df_gt.columns))}")
        st.stop()

    true_labels = df_gt["match"].tolist()
    results = []
    scores = {
        "sbert_cosine": [], "sbert_dot": [],
        "openai_cosine": [], "openai_dot": []
    }

    # Compute embeddings & similarities
    for _, row in df_gt.iterrows():
        k1, k2, match = row["keyword1"], row["keyword2"], row["match"]
        out = {"Keyword 1": k1, "Keyword 2": k2, "Ground Truth Match": match}

        if use_sbert:
            e1 = get_embedding_sbert(k1)
            e2 = get_embedding_sbert(k2)
            out["SBERT Cosine"] = compute_cosine(e1, e2)
            out["SBERT Dot"]    = compute_dot(e1, e2)
            scores["sbert_cosine"].append(out["SBERT Cosine"])
            scores["sbert_dot"].append(out["SBERT Dot"])

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

    # Find best thresholds per metric
    best_thresholds = {}
    for metric, vals in scores.items():
        arr = np.array([v for v in vals if v is not None])
        if arr.size == 0:
            continue
        lb, ub = np.percentile(arr, 10), np.percentile(arr, 90)
        best_f1, best_t = 0, lb
        for t in np.linspace(lb, ub, 50):
            preds = [1 if v >= t else 0 for v in vals]
            p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary')
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[metric] = best_t

    # Build output DataFrame
    df_out = pd.DataFrame(results)

    # Classification errors
    for model in ["SBERT", "OpenAI"]:
        for mtype in ["Cosine", "Dot"]:
            col = f"{model} {mtype}"
            thr = best_thresholds.get(f"{model.lower()}_{mtype.lower()}", 0)
            err_col = f"{col} Error"
            if col in df_out.columns:
                df_out[err_col] = df_out.apply(
                    lambda r: "False Positive"
                              if r[col] is not None and r[col] >= thr and r["Ground Truth Match"] == 0
                              else "False Negative"
                              if r[col] is not None and r[col] <  thr and r["Ground Truth Match"] == 1
                              else "Correct",
                    axis=1
                )

    st.dataframe(df_out)

    # Show thresholds
    st.write("### Recommended Thresholds")
    for k, v in best_thresholds.items():
        st.write(f"- **{k.replace('_',' ').title()}**: {v:.4f}")

    # CSV download
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "similarity_results.csv", "text/csv")
