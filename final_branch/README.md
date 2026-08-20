# Emergency Medical RAG System

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG)
system for answering questions about emergency medical situations
using information extracted from a medical emergency PDF.

The system retrieves relevant document chunks and uses an LLM to
generate an answer grounded in the retrieved source context.

## Pipeline

PDF
↓
Text Extraction
↓
Cleaning
↓
Chunking
↓
Embedding Generation
↓
ChromaDB Vector Database
↓
Semantic Retrieval
↓
Top-K Relevant Chunks
↓
Llama 3.2
↓
Grounded Answer + Source Pages

## Ingestion

The PDF is processed using PyPDF.

Final chunking configuration:

- Chunk size: 300 tokens
- Overlap: 30 tokens

## Embeddings

Embedding model:

all-MiniLM-L6-v2

The generated embeddings are stored in ChromaDB.

## Retrieval

The system uses semantic similarity search with:

- Top-K = 4

Hybrid retrieval using BM25 + Semantic Search + Reciprocal Rank
Fusion (RRF) was also evaluated.

## Retrieval Evaluation

Semantic Search:

P@3 = 0.467
P@4 = 0.475
P@5 = 0.440

Hybrid RRF:

P@3 = 0.533
P@4 = 0.475
P@5 = 0.440

Based on the evaluation, Semantic Search was selected as the
primary retrieval method because it achieved the same P@4 as
Hybrid RRF while keeping the final retrieval pipeline simpler.

## Generation

The retrieved context is passed to Llama 3.2 through Ollama.

The generation prompt instructs the model to:

- Use only the retrieved source context.
- Avoid outside knowledge.
- Avoid inventing information.
- Answer concisely.
- State when the source does not contain enough information.

## Chatbot

Run:

python rag_generate.py

The chatbot accepts multiple questions in one session.

Type:

exit

to stop the chatbot.

## Project Structure

code/
    ingest.py
    query.py
    rag_generate.py
    hybrid_search.py
    hybrid_evaluate.py
    evaluate.py
    embedding_benchmark.py
    final_test_fast.py
    save_results.py

data/
    em.pdf
    test_set.json

results/
    Evaluation and experiment results

## Final System

The final system provides:

- PDF ingestion
- Text chunking
- Embedding generation
- Vector storage
- Semantic retrieval
- Hybrid retrieval evaluation
- LLM-based answer generation
- Source page attribution
- Interactive chatbot