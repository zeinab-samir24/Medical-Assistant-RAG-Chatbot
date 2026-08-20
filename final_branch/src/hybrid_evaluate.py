import json
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

K_VALUES = [3, 4, 5]

CANDIDATE_K = 10

RRF_K = 60

TEST_FILE = "test_set.json"


# =====================================================
# TOKENIZER
# =====================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# =====================================================
# RELEVANCE
# =====================================================

def is_relevant(
    document,
    expected_keywords
):

    document_lower = document.lower()

    matched = 0

    for keyword in expected_keywords:

        if keyword.lower() in document_lower:

            matched += 1

    required = max(
        1,
        len(expected_keywords) // 2
    )

    return matched >= required


# =====================================================
# RRF
# =====================================================

def reciprocal_rank_fusion(
    semantic_ids,
    bm25_ids
):

    scores = {}

    # Semantic ranking
    for rank, doc_id in enumerate(
        semantic_ids,
        start=1
    ):

        scores[doc_id] = (
            scores.get(doc_id, 0)
            +
            1 / (RRF_K + rank)
        )


    # BM25 ranking
    for rank, doc_id in enumerate(
        bm25_ids,
        start=1
    ):

        scores[doc_id] = (
            scores.get(doc_id, 0)
            +
            1 / (RRF_K + rank)
        )


    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc_id
        for doc_id, score in ranked
    ]


# =====================================================
# LOAD TEST SET
# =====================================================

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:

    test_set = json.load(file)


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = SentenceTransformer(
    MODEL_NAME
)


# =====================================================
# CONNECT TO CHROMA
# =====================================================

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =====================================================
# LOAD DOCUMENTS
# =====================================================

data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = data["documents"]

metadatas = data["metadatas"]


print(
    "Documents:",
    len(documents)
)


# =====================================================
# BUILD BM25
# =====================================================

print("Building BM25...")

tokenized_documents = [
    tokenize(document)
    for document in documents
]

bm25 = BM25Okapi(
    tokenized_documents
)


# =====================================================
# RESULTS STORAGE
# =====================================================

semantic_precision = {
    k: []
    for k in K_VALUES
}

hybrid_precision = {
    k: []
    for k in K_VALUES
}


# =====================================================
# EVALUATE QUESTIONS
# =====================================================

for question_number, test in enumerate(
    test_set,
    start=1
):

    question = test["question"]

    expected_keywords = (
        test["expected_keywords"]
    )


    print("\n")
    print("=" * 70)

    print(
        f"Question {question_number}:"
    )

    print(question)

    print("=" * 70)


    # =================================================
    # SEMANTIC SEARCH
    # =================================================

    query_embedding = model.encode(
        [question]
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


    # Map document text to ID
    document_to_id = {
        document: i
        for i, document in enumerate(
            documents
        )
    }


    semantic_ids = []

    for document in semantic_documents:

        doc_id = document_to_id.get(
            document
        )

        if doc_id is not None:

            semantic_ids.append(
                doc_id
            )


    # =================================================
    # BM25 SEARCH
    # =================================================

    query_tokens = tokenize(
        question
    )


    bm25_scores = bm25.get_scores(
        query_tokens
    )


    bm25_ids = sorted(

        range(len(bm25_scores)),

        key=lambda i:
            bm25_scores[i],

        reverse=True

    )[:CANDIDATE_K]


    # =================================================
    # HYBRID / RRF
    # =================================================

    hybrid_ids = reciprocal_rank_fusion(
        semantic_ids,
        bm25_ids
    )


    # =================================================
    # PRECISION@K
    # =================================================

    for k in K_VALUES:

        # -----------------------------
        # Semantic
        # -----------------------------

        semantic_top_k = [
            documents[i]
            for i in semantic_ids[:k]
        ]


        semantic_relevant = sum(

            is_relevant(
                document,
                expected_keywords
            )

            for document
            in semantic_top_k
        )


        semantic_p = (
            semantic_relevant / k
        )


        semantic_precision[k].append(
            semantic_p
        )


        # -----------------------------
        # Hybrid
        # -----------------------------

        hybrid_top_k = [
            documents[i]
            for i in hybrid_ids[:k]
        ]


        hybrid_relevant = sum(

            is_relevant(
                document,
                expected_keywords
            )

            for document
            in hybrid_top_k
        )


        hybrid_p = (
            hybrid_relevant / k
        )


        hybrid_precision[k].append(
            hybrid_p
        )


    # =================================================
    # SHOW TOP 3 HYBRID RESULTS
    # =================================================

    print("\nTop 3 Hybrid Results:")

    for rank, doc_id in enumerate(
        hybrid_ids[:3],
        start=1
    ):

        print(
            f"\n--- Rank {rank} ---"
        )

        print(
            "Page:",
            metadatas[doc_id].get(
                "page",
                "N/A"
            )
        )

        print(
            "Relevant:",
            is_relevant(
                documents[doc_id],
                expected_keywords
            )
        )

        print(
            "Text:",
            documents[doc_id][:300]
        )


# =====================================================
# AVERAGE RESULTS
# =====================================================

semantic_final = {}

hybrid_final = {}


for k in K_VALUES:

    semantic_final[k] = (
        sum(semantic_precision[k])
        /
        len(semantic_precision[k])
    )

    hybrid_final[k] = (
        sum(hybrid_precision[k])
        /
        len(hybrid_precision[k])
    )


# =====================================================
# FINAL COMPARISON
# =====================================================

print("\n\n")

print("=" * 80)

print(
    "              SEMANTIC VS HYBRID"
)

print("=" * 80)

print(
    f"{'Method':<15}"
    f"{'P@3':<15}"
    f"{'P@4':<15}"
    f"{'P@5':<15}"
)

print("-" * 60)


print(
    f"{'Semantic':<15}"
    f"{semantic_final[3]:<15.3f}"
    f"{semantic_final[4]:<15.3f}"
    f"{semantic_final[5]:<15.3f}"
)


print(
    f"{'Hybrid RRF':<15}"
    f"{hybrid_final[3]:<15.3f}"
    f"{hybrid_final[4]:<15.3f}"
    f"{hybrid_final[5]:<15.3f}"
)


# =====================================================
# IMPROVEMENT
# =====================================================

print("\n")

print("=" * 80)

print(
    "                 IMPROVEMENT"
)

print("=" * 80)


for k in K_VALUES:

    improvement = (
        hybrid_final[k]
        -
        semantic_final[k]
    )

    print(
        f"P@{k}: "
        f"{improvement:+.3f}"
    )


# =====================================================
# BEST METHOD
# =====================================================

if hybrid_final[4] > semantic_final[4]:

    best_method = "Hybrid RRF"

else:

    best_method = "Semantic Search"


print("\n")

print(
    "Best method based on P@4:",
    best_method
)