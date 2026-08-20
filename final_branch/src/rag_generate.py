import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# =====================================================
# SETTINGS
# =====================================================

DB_PATH = "./chroma_db_300_30"
COLLECTION_NAME = "medical_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

TOP_K = 4


# =====================================================
# LOAD ONCE
# =====================================================

print("Loading model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Chatbot ready!")
print("Type 'exit' to stop.\n")


# =====================================================
# CHAT LOOP
# =====================================================

while True:

    question = input("Ask a question: ").strip()

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    if not question:
        continue

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

    context = "\n".join(context_parts)

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
4. Answer directly and concisely.
5. Include the important information relevant to the question.
6. Do not mix unrelated information.
7. If the source does not contain enough information to answer
   the question, say:
   "The provided source does not contain enough information
   to answer this question."
8. Do not provide page numbers in the answer.

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
    # ANSWER
    # -------------------------------------------------

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    # -------------------------------------------------
    # SOURCES
    # -------------------------------------------------

    pages = sorted(
        set(
            metadata.get(
                "page",
                "N/A"
            )
            for metadata in metadatas
        )
    )

    print("\nSource pages:", pages)

    print("\n" + "=" * 70 + "\n")