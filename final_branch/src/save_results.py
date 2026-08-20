import csv
import os

os.makedirs("../results", exist_ok=True)


# =====================================================
# CHUNKING EXPERIMENT RESULTS
# =====================================================

chunking_results = [
    ["300/30", 0.467, 0.475, 0.440],
    ["400/50", 0.467, 0.400, 0.360],
    ["500/75", 0.500, 0.450, 0.400],
]

with open(
    "../results/chunking_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "configuration",
        "precision_at_3",
        "precision_at_4",
        "precision_at_5"
    ])

    writer.writerows(chunking_results)


# =====================================================
# EMBEDDING MODEL RESULTS
# =====================================================

embedding_results = [
    [
        "all-MiniLM-L6-v2",
        0.467,
        0.475,
        0.440,
        0.0250
    ],
    [
        "all-mpnet-base-v2",
        0.433,
        0.400,
        0.360,
        0.1275
    ],
]

with open(
    "../results/embedding_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "model",
        "precision_at_3",
        "precision_at_4",
        "precision_at_5",
        "average_query_latency_seconds"
    ])

    writer.writerows(embedding_results)


print("Results saved successfully!")

print(
    "\nCreated:"
)

print(
    "results/chunking_results.csv"
)

print(
    "results/embedding_results.csv"
)