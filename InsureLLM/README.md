# Insurellm RAG Chatbot - Expert Knowledge Worker

This project demonstrates a robust Retrieval-Augmented Generation (RAG) workflow designed to integrate organizational data ("Insurellm") into a conversational AI. It showcases how to build, query, and rigorously evaluate a RAG system using Google's Gemini models and ChromaDB.

App Deployed: https://insurelm-rag-chatbot.streamlit.app/

## 🚀 Key Features

* **RAG Pipeline:** Implements a full retrieval workflow including query rewriting, embedding generation, vector search, and reranking.
* **Dual-Stage Retrieval:** Combines initial vector search (Retrieval-K) with a reranking step (Final-K) to improve context relevance.
* **Query Expansion:** Automatically rewrites user queries to improve search hits in the knowledge base.
* **Quantitative Evaluation:** Includes a custom evaluation framework measuring MRR (Mean Reciprocal Rank), nDCG (Normalized Discounted Cumulative Gain), and Keyword Coverage.
* **Gemini Powered:** Utilizes `gemini-2.5-flash` for generation and `gemini-embedding-001` for vector embeddings.

## 🛠️ Architecture

The system follows this workflow:

1.  **Ingestion:** Documents are preprocessed and stored in a persistent **ChromaDB** vector store.
2.  **Query Processing:**
    * The user's question is rewritten to be more search-friendly.
    * Both the original and rewritten questions are embedded and queried against the database.
3.  **Reranking:** The retrieved chunks are merged and passed to a reranker (LLM-based) to order them by strict relevance.
4.  **Generation:** The top $K$ chunks are fed into the system prompt to generate the final answer.

## 📊 Evaluation Metrics

To ensure the reliability of the retrieval system, the following metrics are tracked:

| Metric | Description |
| :--- | :--- |
| **MRR (Mean Reciprocal Rank)** | Measures how high up the relevant information appears in the retrieval list. Higher scores indicate relevant data is found earlier. |
| **nDCG (Normalized Discounted Cumulative Gain)** | Evaluates the ranking quality, giving more weight to highly relevant documents appearing at the top of the list. |
| **Keyword Coverage** | Calculates the percentage of required keywords (ground truth) that appear in the retrieved context chunks. |

## 💻 Tech Stack

* **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`)
* **Vector Store:** ChromaDB
* **Orchestration:** LangChain (Google GenAI integration) & LiteLLM
* **Resilience:** Tenacity (for retries)
* **Validation:** Pydantic

## 📈 Usage Example

**Query:** *"What is the policy on deductibles?"*

1.  **Rewrite:** System rewrites to *"Insurellm deductible policy details"*
2.  **Retrieval:** Fetches top 20 chunks from ChromaDB.
3.  **Rerank:** Sorts chunks and selects top 10.
4.  **Response:** Generates a precise answer based on the knowledge base.