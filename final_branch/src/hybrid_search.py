import re

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# =====================================================
# SETTINGS
# =====================================================

DB_PATH = "./chroma_db_300_30"

COLLECTION_NAME = "medical_knowledge"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 4

# We retrieve more candidates first,
# then combine them using RRF.
CANDIDATE_K = 10

# RRF constant
RRF_K = 60


# =====================================================
# TOKENIZER
# =====================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# =====================================================
# CONNECT TO CHROMADB
# =====================================================

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =====================================================
# LOAD DOCUMENTS FOR BM25
# =====================================================

print("Loading documents...")

data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = data["documents"]

metadatas = data["metadatas"]


print(
    "Documents loaded:",
    len(documents)
)


# =====================================================
# BUILD BM25 INDEX
# =====================================================

print("Building BM25 index...")

tokenized_documents = [
    tokenize(document)
    for document in documents
]

bm25 = BM25Okapi(
    tokenized_documents
)

print("BM25 ready!")


# =====================================================
# RRF FUNCTION
# =====================================================

def reciprocal_rank_fusion(
    semantic_ranked,
    bm25_ranked
):

    scores = {}

    # -----------------------------------------------
    # Semantic ranking
    # -----------------------------------------------

    for rank, doc_id in enumerate(
        semantic_ranked,
        start=1
    ):

        scores[doc_id] = (
            scores.get(doc_id, 0)
            +
            1 / (RRF_K + rank)
        )


    # -----------------------------------------------
    # BM25 ranking
    # -----------------------------------------------

    for rank, doc_id in enumerate(
        bm25_ranked,
        start=1
    ):

        scores[doc_id] = (
            scores.get(doc_id, 0)
            +
            1 / (RRF_K + rank)
        )


    # -----------------------------------------------
    # Sort by RRF score
    # -----------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# =====================================================
# QUERY
# =====================================================

query = input(
    "\nAsk a question: "
).strip()


if not query:

    print("Please enter a question.")

    exit()


# =====================================================
# 1. SEMANTIC SEARCH
# =====================================================

query_embedding = model.encode(
    [query]
)


semantic_results = collection.query(

    query_embeddings=
    query_embedding.tolist(),

    n_results=CANDIDATE_K,

    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


semantic_documents = (
    semantic_results["documents"][0]
)

semantic_metadatas = (
    semantic_results["metadatas"][0]
)

semantic_distances = (
    semantic_results["distances"][0]
)


# Map document text to database index
document_to_id = {
    document: i
    for i, document in enumerate(documents)
}


# Semantic ranking
semantic_ranked_ids = []

semantic_distance_map = {}


for document, distance in zip(
    semantic_documents,
    semantic_distances
):

    doc_id = document_to_id.get(
        document
    )

    if doc_id is not None:

        semantic_ranked_ids.append(
            doc_id
        )

        semantic_distance_map[
            doc_id
        ] = distance


# =====================================================
# 2. BM25 SEARCH
# =====================================================

query_tokens = tokenize(
    query
)


bm25_scores = bm25.get_scores(
    query_tokens
)


bm25_ranked_ids = sorted(

    range(len(bm25_scores)),

    key=lambda i:
        bm25_scores[i],

    reverse=True
)[:CANDIDATE_K]


# =====================================================
# 3. RRF
# =====================================================

hybrid_ranked = reciprocal_rank_fusion(

    semantic_ranked_ids,

    bm25_ranked_ids

)


# =====================================================
# FINAL TOP-K
# =====================================================

final_results = hybrid_ranked[:TOP_K]


# =====================================================
# EVIDENCE VIEW
# =====================================================

print("\n")
print("=" * 75)

print(
    "                    HYBRID RETRIEVAL"
)

print("=" * 75)

print("\nQuestion:")

print(query)


for rank, (
    doc_id,
    rrf_score
) in enumerate(
    final_results,
    start=1
):

    print("\n")
    print("-" * 75)

    print(
        f"Rank: {rank}"
    )

    print(
        "RRF Score:",
        round(
            rrf_score,
            6
        )
    )

    print(
        "Semantic Distance:",
        round(
            semantic_distance_map.get(
                doc_id,
                -1
            ),
            6
        )
    )

    print(
        "BM25 Score:",
        round(
            bm25_scores[doc_id],
            6
        )
    )

    print(
        "Page:",
        metadatas[doc_id].get(
            "page",
            "N/A"
        )
    )

    print(
        "Source:",
        metadatas[doc_id].get(
            "source",
            "N/A"
        )
    )

    print("\nRetrieved Text:")

    print(
        documents[doc_id]
    )


print("\n")
print("=" * 75)

print(
    "                  END OF HYBRID SEARCH"
)

print("=" * 75)