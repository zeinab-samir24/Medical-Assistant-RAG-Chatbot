import json
import re
import time

import chromadb
from sentence_transformers import SentenceTransformer


# =====================================================
# SETTINGS
# =====================================================

CHUNK_DB_PATH = "./chroma_db_300_30"

COLLECTION_NAME = "medical_knowledge"

TEST_FILE = "test_set.json"

MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]

K_VALUES = [3, 4, 5]


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
# LOAD TEST SET
# =====================================================

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:

    test_set = json.load(file)


# =====================================================
# LOAD EXISTING CHUNKS
# =====================================================

print("\nLoading chunks...")

client = chromadb.PersistentClient(
    path=CHUNK_DB_PATH
)

old_collection = client.get_collection(
    name=COLLECTION_NAME
)

data = old_collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = data["documents"]
metadatas = data["metadatas"]

print(
    "Number of chunks:",
    len(documents)
)


# =====================================================
# BENCHMARK
# =====================================================

all_results = {}


for model_name in MODELS:

    print("\n")
    print("=" * 70)

    print(
        "MODEL:",
        model_name
    )

    print("=" * 70)


    # -----------------------------------------------
    # Load model
    # -----------------------------------------------

    start_model = time.perf_counter()

    model = SentenceTransformer(
        model_name
    )

    model_load_time = (
        time.perf_counter()
        - start_model
    )


    # -----------------------------------------------
    # Create embeddings for chunks
    # -----------------------------------------------

    print(
        "\nCreating document embeddings..."
    )

    start_embedding = time.perf_counter()

    document_embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    embedding_time = (
        time.perf_counter()
        - start_embedding
    )


    # -----------------------------------------------
    # Create temporary collection
    # -----------------------------------------------

    safe_name = (
        model_name
        .replace("-", "_")
        .replace(".", "_")
    )

    collection_name = (
        "benchmark_" + safe_name
    )


    # Delete if already exists

    try:

        client.delete_collection(
            name=collection_name
        )

    except Exception:

        pass


    benchmark_collection = (
        client.create_collection(
            name=collection_name
        )
    )


    # -----------------------------------------------
    # Store embeddings
    # -----------------------------------------------

    benchmark_collection.add(

        ids=[
            f"doc_{i}"
            for i in range(len(documents))
        ],

        documents=documents,

        embeddings=document_embeddings.tolist(),

        metadatas=metadatas
    )


    # -----------------------------------------------
    # Evaluation
    # -----------------------------------------------

    precision_results = {
        k: []
        for k in K_VALUES
    }

    query_times = []


    for test in test_set:

        question = test["question"]

        expected_keywords = (
            test["expected_keywords"]
        )


        # -------------------------------------------
        # Query embedding
        # -------------------------------------------

        start_query = time.perf_counter()

        query_embedding = model.encode(
            [question]
        )


        results = benchmark_collection.query(

            query_embeddings=
            query_embedding.tolist(),

            n_results=max(K_VALUES),

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )


        query_time = (
            time.perf_counter()
            - start_query
        )

        query_times.append(
            query_time
        )


        retrieved_documents = (
            results["documents"][0]
        )


        # -------------------------------------------
        # Precision@K
        # -------------------------------------------

        for k in K_VALUES:

            top_documents = (
                retrieved_documents[:k]
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


    # -----------------------------------------------
    # Average Precision
    # -----------------------------------------------

    final_precision = {}


    for k in K_VALUES:

        values = precision_results[k]

        final_precision[k] = (
            sum(values)
            / len(values)
        )


    average_query_time = (
        sum(query_times)
        / len(query_times)
    )


    all_results[model_name] = {

        "P@3":
            final_precision[3],

        "P@4":
            final_precision[4],

        "P@5":
            final_precision[5],

        "query_latency":
            average_query_time,

        "model_load_time":
            model_load_time,

        "embedding_time":
            embedding_time
    }


    print("\nResults for:", model_name)

    print(
        "Precision@3:",
        round(
            final_precision[3],
            3
        )
    )

    print(
        "Precision@4:",
        round(
            final_precision[4],
            3
        )
    )

    print(
        "Precision@5:",
        round(
            final_precision[5],
            3
        )
    )

    print(
        "Average query latency:",
        round(
            average_query_time,
            4
        ),
        "seconds"
    )


# =====================================================
# FINAL COMPARISON
# =====================================================

print("\n\n")

print("=" * 85)

print(
    "                 EMBEDDING MODEL BENCHMARK"
)

print("=" * 85)

print(
    f"{'Model':<30}"
    f"{'P@3':<12}"
    f"{'P@4':<12}"
    f"{'P@5':<12}"
    f"{'Latency':<12}"
)

print("-" * 85)


for model_name, result in all_results.items():

    print(
        f"{model_name:<30}"
        f"{result['P@3']:<12.3f}"
        f"{result['P@4']:<12.3f}"
        f"{result['P@5']:<12.3f}"
        f"{result['query_latency']:<12.4f}"
    )


# =====================================================
# BEST MODEL
# =====================================================

best_model = max(
    all_results,
    key=lambda model:
        all_results[model]["P@4"]
)


print("\n")
print("=" * 85)

print(
    "BEST MODEL BASED ON PRECISION@4:"
)

print(
    best_model
)

print(
    "Precision@4:",
    round(
        all_results[
            best_model
        ]["P@4"],
        3
    )
)

print("=" * 85)