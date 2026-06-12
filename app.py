import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ==========================================
# 1. INITIALIZATION & DATABASE CONNECTION
# ==========================================
print("[INFO] Loading local HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("[INFO] Connecting to local ChromaDB...")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Configure the retriever to fetch the top 4 most relevant chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

print("[INFO] Connecting to local Gemma 4 model...")
llm = OllamaLLM(
    model="gemma4:e4b", 
    temperature=0.1,
    base_url="http://localhost:11434"
)

# ==========================================
# 2. RAG PIPELINE CONSTRUCTION (LCEL)
# ==========================================
# Define strict instructions to prevent hallucination
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based ONLY on the provided context. 
If the answer is not in the context, say "I cannot find the answer in the dataset."

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    """Formats retrieved document chunks into a single readable string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the internal processing chain
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_template
    | llm
    | StrOutputParser()
)

# Build the parallel chain to output both the LLM answer and the source documents
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# ==========================================
# 3. INTERACTIVE CLI CHATBOT LOOP
# ==========================================
def interactive_chat():
    """
    Runs an interactive command-line chat loop.
    Allows dynamic user inputs and calculates performance metrics per query.
    """
    print("\n" + "="*50)
    print("🎬 MOVIE RAG ASSISTANT IS READY!")
    print("Type your questions below. Type 'quit', 'exit', or 'q' to stop.")
    print("="*50 + "\n")
    
    # Infinite loop to keep the chatbot running
    while True:
        # 1. Get dynamic user input
        user_query = input("\n[YOU]: ").strip()
        
        # 2. Check for exit commands
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n[INFO] Exiting the RAG Assistant. Goodbye!")
            break
            
        # Ignore empty inputs
        if not user_query:
            continue
            
        print("-" * 50)
        
        # 3. Start timer for latency evaluation
        start_time = time.time()
        
        try:
            # Invoke the LCEL pipeline with the user's dynamic query
            response = rag_chain_with_source.invoke(user_query)
        except Exception as e:
            print(f"[ERROR] LLM Generation failed: {e}")
            continue
            
        # 4. Calculate execution time
        latency = round(time.time() - start_time, 2)
        
        # 5. Display the AI response, metrics, and sources
        print(f"\n[AI RESPONSE]:\n{response['answer']}\n")
        print(f"[METRIC] Processing Latency: {latency} seconds")
        print("[RETRIEVED SOURCES]:")
        
        for idx, doc in enumerate(response["context"]):
            print(f"  {idx + 1}. {doc.metadata['title']} ({doc.metadata['year']})")

if __name__ == "__main__":
    # Start the interactive loop
    interactive_chat()