import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import fasttext
import cohere
import os

# Load models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
fasttext_model = fasttext.load_model('cc.en.300.bin')  # Make sure to have this file downloaded
cohere_client = cohere.Client("YOUR_COHERE_API_KEY")  # Replace with your key

def get_embedding_sbert(text):
    return sbert_model.encode(text, normalize_embeddings=True)

def get_embedding_openai(text, api_key):
    openai.api_key = api_key
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

def get_embedding_fasttext(text):
    words = text.split()
    word_vectors = [fasttext_model.get_word_vector(word) for word in words]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)

def get_embedding_cohere(text):
    response = cohere_client.embed(model='embed-english-light-v2.0', texts=[text])
    return response.embeddings[0]

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
use_fasttext = st.checkbox("Use FastText", True)
use_cohere = st.checkbox("Use Cohere", True)

if st.button("Compute Similarity"):
    results = []
    for k1 in keyword_list_1:
        for k2 in keyword_list_2:
            row = [k1, k2]
            if use_sbert:
                row.append(compute_similarity(get_embedding_sbert(k1), get_embedding_sbert(k2)))
            if use_openai and api_key:
                row.append(compute_similarity(get_embedding_openai(k1, api_key), get_embedding_openai(k2, api_key)))
            if use_fasttext:
                row.append(compute_similarity(get_embedding_fasttext(k1), get_embedding_fasttext(k2)))
            if use_cohere:
                row.append(compute_similarity(get_embedding_cohere(k1), get_embedding_cohere(k2)))
            results.append(row)
    
    # Dataframe Output
    columns = ["Keyword 1", "Keyword 2"]
    if use_sbert:
        columns.append("SBERT Similarity")
    if use_openai and api_key:
        columns.append("OpenAI Similarity")
    if use_fasttext:
        columns.append("FastText Similarity")
    if use_cohere:
        columns.append("Cohere Similarity")
    
    df = pd.DataFrame(results, columns=columns)
    st.dataframe(df)
    
    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "similarity_results.csv", "text/csv")
