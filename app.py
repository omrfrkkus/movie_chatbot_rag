"""
app.py

Interactive Retrieval-Augmented Generation (RAG) CLI.
Utilizes LangChain Expression Language (LCEL) to orchestrate a local vector store 
(ChromaDB) and a local LLM via Ollama. Designed for zero-hallucination querying.
"""

import time
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# --- Configuration & Initialization ---
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434"
RETRIEVER_K = 4

print("[INFO] Initializing embedding model and connecting to vector store...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

print(f"[INFO] Establishing connection to local LLM ({LLM_MODEL})...")
llm = OllamaLLM(
    model=LLM_MODEL, 
    temperature=0.1,  # Low temperature to prioritize factual retrieval
    base_url=OLLAMA_BASE_URL
)

# --- LCEL Pipeline Construction ---

# Strict prompt design to prevent out-of-domain hallucinations
RAG_TEMPLATE = """
Answer the following question based ONLY on the provided context. 
If the answer is not in the context, say "I cannot find the answer in the dataset."

Context:
{context}

Question:
{question}
"""
prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)

def format_docs(docs: List[Document]) -> str:
    """Aggregates retrieved document contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Internal processing chain: Context formatting -> Prompt -> LLM -> String Output
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_template
    | llm
    | StrOutputParser()
)

# Final parallel chain: Retains source documents alongside the generated answer
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# --- Interactive CLI ---

def interactive_chat() -> None:
    """
    Initializes the command-line interface for real-time querying.
    Captures execution latency and retrieval metadata for performance auditing.
    """
    print("\n" + "=" * 50)
    print("🎬 CINEMATIC RAG ENGINE: ONLINE")
    print("Awaiting queries. Type 'quit', 'exit', or 'q' to terminate session.")
    print("=" * 50 + "\n")
    
    while True:
        try:
            user_query = input("\n[USER]: ").strip()
        except KeyboardInterrupt:
            break
            
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n[SYSTEM] Terminating session. Goodbye!")
            break
            
        if not user_query:
            continue
            
        print("-" * 50)
        start_time = time.time()
        
        try:
            # Execute the LCEL pipeline
            response = rag_chain_with_source.invoke(user_query)
        except Exception as e:
            print(f"[ERROR] Pipeline execution failed: {e}")
            continue
            
        latency = round(time.time() - start_time, 2)
        
        # Output formulation
        print(f"\n[AI]:\n{response['answer']}\n")
        print(f"[METRICS] Execution Latency: {latency}s")
        print("[SOURCES]:")
        
        for idx, doc in enumerate(response["context"], 1):
            title = doc.metadata.get('title', 'Unknown Title')
            year = doc.metadata.get('year', 'Unknown Year')
            print(f"  {idx}. {title} ({year})")

if __name__ == "__main__":
    interactive_chat()