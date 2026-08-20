import json
import re

from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi


# =====================================================
# SETTINGS
# =====================================================

MODEL_NAME = "all-MiniLM-L6-v2"

CONFIGURATIONS = [
    {
        "name": "300/30",
        "db_path": "./chroma_db_300_30"
    },
    {
        "name": "400/50",
        "db_path": "./chroma_db_400_50"
    },
    {
        "name": "500/75",
        "db_path": "./chroma_db_500_75"
    }
]

K_VALUES = [3, 4, 5]

TEST_FILE = "test_set.json"

COLLECTION_NAME = "medical_knowledge"


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

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# =====================================================
# TOKENIZER
# =====================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# =====================================================
# CHECK RELEVANCE
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

    # At least half of the expected
    # evidence should appear.

    required = max(
        1,
        len(expected_keywords) // 2
    )

    return matched >= required


# =====================================================
# EVALUATE ONE CONFIGURATION
# =====================================================

def evaluate_configuration(
    config
):

    print("\n")
    print("=" * 70)

    print(
        "CONFIGURATION:",
        config["name"]
    )

    print("=" * 70)


    # -----------------------------------------------
    # Connect to database
    # -----------------------------------------------

    client = chromadb.PersistentClient(
        path=config["db_path"]
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )


    # -----------------------------------------------
    # Load documents
    # -----------------------------------------------

    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = data["documents"]

    metadatas = data["metadatas"]


    print(
        "Chunks:",
        len(documents)
    )


    # -----------------------------------------------
    # Prepare BM25
    # -----------------------------------------------

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )


    # -----------------------------------------------
    # Store results
    # -----------------------------------------------

    precision_results = {
        k: []
        for k in K_VALUES
    }


    # =================================================
    # QUESTIONS
    # =================================================

    for test in test_set:

        question = test["question"]

        expected_keywords = (
            test["expected_keywords"]
        )


        print("\n")
        print(
            "Question:",
            question
        )


        # ---------------------------------------------
        # Semantic search
        # ---------------------------------------------

        query_embedding = model.encode(
            [question]
        )


        semantic_results = collection.query(

            query_embeddings=
            query_embedding.tolist(),

            n_results=max(K_VALUES),

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )


        semantic_documents = (
            semantic_results["documents"][0]
        )


        semantic_distances = (
            semantic_results["distances"][0]
        )


        # ---------------------------------------------
        # Evaluate each K
        # ---------------------------------------------

        for k in K_VALUES:

            top_documents = (
                semantic_documents[:k]
            )

            relevant_count = 0


            for document in top_documents:

                if is_relevant(
                    document,
                    expected_keywords
                ):

                    relevant_count += 1


            precision = (
                relevant_count / k
            )


            precision_results[k].append(
                precision
            )


        # ---------------------------------------------
        # Show retrieved evidence
        # ---------------------------------------------

        print("\nTop 3 retrieved evidence:")


        for rank, document in enumerate(
            semantic_documents[:3],
            start=1
        ):

            print(
                f"\n--- Rank {rank} ---"
            )

            print(
                "Page:",
                semantic_results[
                    "metadatas"
                ][0][rank - 1]["page"]
            )

            print(
                "Distance:",
                round(
                    semantic_distances[
                        rank - 1
                    ],
                    5
                )
            )

            print(
                "Relevant:",
                is_relevant(
                    document,
                    expected_keywords
                )
            )

            print(
                "Text:",
                document[:500]
            )


    # =================================================
    # AVERAGE PRECISION
    # =================================================

    final_results = {}


    for k in K_VALUES:

        values = precision_results[k]

        average_precision = (
            sum(values)
            / len(values)
        )

        final_results[k] = (
            average_precision
        )


    return final_results


# =====================================================
# RUN ALL CONFIGURATIONS
# =====================================================

all_results = {}


for config in CONFIGURATIONS:

    try:

        results = evaluate_configuration(
            config
        )

        all_results[
            config["name"]
        ] = results

    except Exception as error:

        print(
            "\nERROR with configuration:",
            config["name"]
        )

        print(error)


# =====================================================
# FINAL TABLE
# =====================================================

print("\n\n")
print("=" * 70)

print(
    "             PRECISION@K RESULTS"
)

print("=" * 70)

print(
    f"{'Config':<15}"
    f"{'P@3':<15}"
    f"{'P@4':<15}"
    f"{'P@5':<15}"
)

print("-" * 60)


for config_name, results in all_results.items():

    print(
        f"{config_name:<15}"
        f"{results.get(3, 0):<15.3f}"
        f"{results.get(4, 0):<15.3f}"
        f"{results.get(5, 0):<15.3f}"
    )


# =====================================================
# FIND BEST CONFIGURATION
# =====================================================

print("\n")
print("=" * 70)

print(
    "             BEST CONFIGURATION"
)

print("=" * 70)


best_config = None
best_score = -1


for config_name, results in all_results.items():

    # We use P@4 as the main comparison
    score = results.get(4, 0)

    if score > best_score:

        best_score = score

        best_config = config_name


print(
    "Best configuration:",
    best_config
)

print(
    "Best Precision@4:",
    round(
        best_score,
        3
    )
)