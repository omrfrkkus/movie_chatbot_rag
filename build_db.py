"""
build_db.py

Responsible for data ingestion, preprocessing, and vector database construction.
Filters a cinematic dataset and generates embeddings using a local HuggingFace model
to ensure offline capabilities and privacy.
"""

import os
import shutil
import pandas as pd

from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration Constants ---
DATASET_PATH = "wiki_movie_plots_deduped.csv"
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Temporal filtering to optimize local vector search space
FILTER_START_YEAR = 1995
FILTER_END_YEAR = 2005

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def build_vector_database() -> None:
    """
    Executes the ETL pipeline: loads data, applies temporal filtering, 
    chunks text, and persists embeddings to a local Chroma database.
    """
    print("[INFO] Initializing dataset ingestion...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Required dataset '{DATASET_PATH}' not found in the root directory.")
        return
        
    df_all = pd.read_csv(DATASET_PATH)
    
    # Isolate records within the specified temporal threshold
    df = df_all[(df_all['Release Year'] >= FILTER_START_YEAR) & (df_all['Release Year'] <= FILTER_END_YEAR)]
    print(f"[INFO] Applied temporal filter ({FILTER_START_YEAR}-{FILTER_END_YEAR}). Retained {len(df)} records.")

    print("[INFO] Transforming dataframe into LangChain Document objects...")
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "title": row['Title'], 
            "year": row['Release Year']
        }
        content = f"Title: {row['Title']}\nPlot: {row['Plot']}"
        documents.append(Document(page_content=content, metadata=metadata))

    print("[INFO] Executing recursive character splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    # Ensure a clean database state to prevent vector duplication
    if os.path.exists(DB_DIR):
        print("[INFO] Purging stale ChromaDB state...")
        shutil.rmtree(DB_DIR)

    print(f"[INFO] Generating embeddings for {len(chunks)} chunks. This may take a moment...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print(f"[SUCCESS] Vector database successfully persisted to {DB_DIR}")

if __name__ == "__main__":
    build_vector_database()