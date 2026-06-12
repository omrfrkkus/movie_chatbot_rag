import pandas as pd
import os
import shutil

from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_database():
    """
    Reads the movie dataset, filters for movies released strictly between 1995 and 2005, 
    splits them into chunks, and embeds them into a local ChromaDB instance
    using local HuggingFace embeddings.
    """
    print("[INFO] Loading CSV dataset...")
    
    if not os.path.exists("wiki_movie_plots_deduped.csv"):
        print("[ERROR] CSV file not found! Please place it in the same directory.")
        return
        
    # Read the entire dataset
    df_all = pd.read_csv("wiki_movie_plots_deduped.csv")
    
    # STRATEGY 1 HIGHLY OPTIMIZED: Filter for movies released between 1995 and 2005
    df = df_all[(df_all['Release Year'] >= 1995) & (df_all['Release Year'] <= 2005)]
    print(f"[INFO] Filtered dataset for movies between 1995 - 2005: {len(df)} movies found.")

    print("[INFO] Converting data into LangChain documents...")
    documents = []
    for _, row in df.iterrows():
        metadata = {"title": row['Title'], "year": row['Release Year']}
        content = f"Title: {row['Title']}\nPlot: {row['Plot']}"
        documents.append(Document(page_content=content, metadata=metadata))

    print("[INFO] Splitting texts into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Prevent database duplication if the script is run multiple times
    if os.path.exists("./chroma_db"):
        print("[INFO] Clearing existing ChromaDB folder to prevent duplicate vectors...")
        shutil.rmtree("./chroma_db")

    print(f"[INFO] Generating embeddings for {len(chunks)} chunks and saving to ChromaDB...")
    
    # Initialize the local HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist the vector database
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    
    print("[SUCCESS] Vector database created successfully!")

if __name__ == "__main__":
    build_vector_database()