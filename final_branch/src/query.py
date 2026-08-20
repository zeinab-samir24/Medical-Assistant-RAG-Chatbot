from sentence_transformers import SentenceTransformer
import chromadb


# =================================
# 1. LOAD EMBEDDING MODEL
# =================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =================================
# 2. CONNECT TO EXISTING CHROMADB
# =================================

client = chromadb.PersistentClient(
    path="./chroma_db_300_30"
)


collection = client.get_collection(
    name="medical_knowledge"
)


# =================================
# 3. ASK QUESTION
# =================================

query = input("\nAsk a question: ")


# =================================
# 4. EMBEDDING FOR QUESTION
# =================================

query_embedding = model.encode([query])


# =================================
# 5. RETRIEVAL
# =================================

results = collection.query(
    query_embeddings=query_embedding.tolist(),

    # نجيب أقرب 4 chunks فقط
    n_results=4
)


# =================================
# 6. DISPLAY RESULTS
# =================================

print("\n===== RETRIEVED RESULTS =====")


for i, document in enumerate(
    results["documents"][0]
):

    print(
        f"\n--- Result {i + 1} ---"
    )

    print("Page:")

    print(
        results["metadatas"][0][i]["page"]
    )

    print("Source:")

    print(
        results["metadatas"][0][i]["source"]
    )

    print("\nText:")

    print(document)
    # =====================================================
# FINAL RETRIEVAL
# =====================================================

TOP_K = 4

query = input("\nAsk a question: ").strip()

if not query:
    print("Please enter a question.")
    exit()


# -----------------------------------------------------
# Create query embedding
# -----------------------------------------------------

query_embedding = model.encode(
    [query]
)


# -----------------------------------------------------
# Semantic Search
# -----------------------------------------------------

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=TOP_K,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# =====================================================
# EVIDENCE VIEW
# =====================================================

print("\n")
print("=" * 70)
print("                 RETRIEVED EVIDENCE")
print("=" * 70)

print("\nQuestion:")
print(query)


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


for i in range(len(documents)):

    print("\n" + "-" * 70)

    print(f"Rank: {i + 1}")

    print(
        "Similarity distance:",
        round(distances[i], 4)
    )

    print(
        "Page:",
        metadatas[i].get("page", "N/A")
    )

    print(
        "Source:",
        metadatas[i].get("source", "N/A")
    )

    print("\nRetrieved Text:")

    print(documents[i])


print("\n" + "=" * 70)
print("                 END OF EVIDENCE")
print("=" * 70)