import chromadb
import ollama

from sentence_transformers import SentenceTransformer


# =====================================================
# SETTINGS
# =====================================================

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "chroma_db_500_75")
COLLECTION_NAME = "medical_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

TOP_K = 4


# =====================================================
# QUESTIONS
# =====================================================

QUESTIONS = [
    "What should you do during a seizure?",
    "When should you call EMS/9-1-1 for a seizure?",
    "What are common signs and symptoms of seizures?",
    "What information does the source provide about seizures that is not covered?"
]


# =====================================================
# LOAD ONCE
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Ready!\n")


# =====================================================
# TEST LOOP
# =====================================================

for test_number, question in enumerate(
    QUESTIONS,
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"TEST {test_number}")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    # -------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------

    query_embedding = model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # -------------------------------------------------
    # BUILD CONTEXT
    # -------------------------------------------------

    context_parts = []

    for i in range(len(documents)):

        page = metadatas[i].get(
            "page",
            "N/A"
        )

        source = metadatas[i].get(
            "source",
            "N/A"
        )

        context_parts.append(
            f"""
SOURCE {i + 1}
Page: {page}
Source: {source}

{documents[i]}
"""
        )

    context = "\n".join(
        context_parts
    )

    # -------------------------------------------------
    # PROMPT
    # -------------------------------------------------

    prompt = f"""
You are a medical information assistant.

Answer the user's question using ONLY the SOURCE CONTEXT.

RULES:

1. Use only information explicitly supported by the source.
2. Do not use outside medical knowledge.
3. Do not invent or assume information.
4. Answer the question directly and concisely.
5. Include the relevant points from the source that directly
   answer the question.
6. Do not mix unrelated information such as causes, symptoms,
   or treatment unless it directly answers the question.
7. ONLY if the source truly does not contain enough information
   to answer the question, say:
   "The provided source does not contain enough information
   to answer this question."
8. Do NOT say that the source is insufficient after already
   providing a complete answer.
9. Do NOT provide page numbers in your answer.
10. Python will provide the source pages separately.

USER QUESTION:
{question}

SOURCE CONTEXT:
{context}
"""

    # -------------------------------------------------
    # GENERATION
    # -------------------------------------------------

    print("\nGenerating...\n")

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    print("-" * 70)
    print("ANSWER:")
    print(answer)

    # -------------------------------------------------
    # SOURCE PAGES
    # -------------------------------------------------

    print("\nPages retrieved:")

    pages = sorted(
        set(
            metadata.get(
                "page",
                "N/A"
            )
            for metadata in metadatas
        )
    )

    print(pages)


# =====================================================
# FINISHED
# =====================================================

print("\n")
print("=" * 70)
print("FINAL TEST COMPLETED")
print("=" * 70)