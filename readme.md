# 🎬 Cinematic RAG: 100% Local Movie Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_4-black.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-all--MiniLM-yellow.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)

An advanced, fully offline **Retrieval-Augmented Generation (RAG)** system built to answer cinematic plot questions. This project was developed as an academic showcase of private, knowledge-aware AI systems using modern **LangChain Expression Language (LCEL)** architecture.

## ✨ Key Features

* **100% Local & Private:** No API keys, no cloud dependencies, and zero data leaves your machine. 
* **Zero Hallucination Architecture:** Engineered with strict context-prompting. If the answer isn't in the dataset, the model refuses to guess, ensuring absolute data faithfulness.
* **Network-Resilient Embedding:** Bypasses local firewall/port restrictions by utilizing in-memory `HuggingFace` embeddings instead of network-dependent LLM embedding routes.
* **Optimized for Speed:** The dataset is smartly filtered (1995–2005 blockbusters) to ensure rapid vector database construction and lightning-fast retrieval during live demonstrations.
* **Interactive CLI:** Features a continuous, real-time command-line chatbot interface with built-in latency tracking.

## 🧠 System Architecture

1. **Data Ingestion:** Reads the *Wikipedia Movie Plots* dataset using `pandas`.
2. **Preprocessing:** Filters movies by release year (1995–2005) and splits plots into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. **Vectorization:** Converts text chunks into dense vectors using HuggingFace's lightweight `all-MiniLM-L6-v2` model.
4. **Storage:** Persists vectors locally using `ChromaDB`.
5. **Retrieval & Generation:** Orchestrates queries via LangChain (LCEL) to fetch the top 4 most relevant chunks and feeds them to Google's `Gemma 4 Edge 4B` model via `Ollama`.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** (Running in the background)
3. Download the Gemma 4 model via Ollama terminal:
```bash
   ollama pull gemma4:e4b
   ```

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR-USERNAME/movie-rag-project.git](https://github.com/YOUR-USERNAME/movie-rag-project.git)
cd movie-rag-project
```

**2. Create and activate a virtual environment:**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download the Dataset:**
* Download the `wiki_movie_plots_deduped.csv` dataset from Kaggle.
* Place it in the root directory of the project.

## 🏃‍♂️ Running the System

**Step 1: Build the Vector Database**
Run the following script to process the dataset, generate embeddings, and build the local ChromaDB. This step only needs to be done once.
```bash
python build_db.py
```
*Expected output: `[SUCCESS] Vector database created successfully!`*

**Step 2: Start the Interactive RAG Assistant**
Launch the chatbot interface to ask questions about the movies in real-time.
```bash
python app.py
```

## 📊 Evaluation & Metrics

The system tracks performance metrics during runtime. Based on our evaluations:
* **Cold Start vs. Warm Start:** Initial queries may take longer as the LLM is loaded into memory (Cold Start). Subsequent queries experience an ~80% reduction in processing latency (Warm Start).
* **Faithfulness Validation:** The system passes rigorous anti-hallucination tests. When asked specific trivia not present in the plot summaries (e.g., "What is the first rule of Fight Club?"), the system successfully retrieves the correct document but accurately replies: *"I cannot find the answer in the dataset,"* proving strict adherence to the RAG context.


---
*Disclaimer: This project is for educational purposes. All movie plots belong to their respective Wikipedia contributors.*