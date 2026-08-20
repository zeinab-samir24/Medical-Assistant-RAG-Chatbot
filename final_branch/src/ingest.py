import re
import os
import shutil
import tiktoken
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
import chromadb

# =====================================================
# SETTINGS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILE = os.path.join(BASE_DIR, "..", "data", "em.pdf")

CHUNK_SIZE = 500
OVERLAP_SIZE = 75

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DB_PATH = os.path.join(BASE_DIR, "..", f"chroma_db_{CHUNK_SIZE}_{OVERLAP_SIZE}")
COLLECTION_NAME = "medical_knowledge"


# =====================================================
# ADD YOUR CUSTOM QUESTIONS & ANSWERS HERE
# =====================================================

CUSTOM_QA_DATA = [
    {
        "question": "What should I do for an asthma attack?",
        "answer": "Help the person sit upright. Assist them in using their rescue inhaler (usually blue). Give 1 puff every 30-60 seconds, up to 10 puffs if needed. Call emergency services if symptoms worsen.",
        "source": "Custom FAQ",
        "page": "FAQ-01"
    },
    {
        "question": "How do I treat severe allergic reaction or anaphylaxis?",
        "answer": "If an EpiPen (epinephrine auto-injector) is available, inject it immediately into the outer thigh. Call 911/112 right away and keep the person calm and lying flat.",
        "source": "Custom FAQ",
        "page": "FAQ-02"
    },
    {
        "question": "What are the first aid steps for a snake bite?",
        "answer": "Keep the person calm and still to slow the spread of venom. Remove tight clothing or jewelry near the bite. Wash the area with soap and water, cover with a clean bandage, and keep the bite location below heart level. Do NOT cut the wound or try to suck out venom.",
        "source": "Custom FAQ",
        "page": "FAQ-03"
    },
     {
            "question": "Do you love me?",
            "answer": "yes, i love you sawsan.",
            "source": "zainab samir",
            "page": "FAQ-03"
        }
    # Add more question/answer dictionaries above as needed!
]


# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =====================================================
# SPLIT SENTENCES
# =====================================================

def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# =====================================================
# CREATE CHUNKS
# =====================================================

def create_chunks(page_text, page_number):
    page_text = clean_text(page_text)
    paragraphs = re.split(r"\n\s*\n", page_text)

    sentences = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_sentences = split_into_sentences(paragraph)
        sentences.extend(paragraph_sentences)

    encoding = tiktoken.get_encoding("cl100k_base")

    chunks = []
    current_sentences = []
    current_tokens = []

    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)

        if len(current_tokens) + len(sentence_tokens) <= CHUNK_SIZE:
            current_sentences.append(sentence)
            current_tokens.extend(sentence_tokens)
        else:
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text": chunk_text,
                    "page": str(page_number),
                    "source": PDF_FILE
                })

            overlap_tokens = current_tokens[-OVERLAP_SIZE:]
            overlap_text = encoding.decode(overlap_tokens)
            overlap_sentences = split_into_sentences(overlap_text)

            current_sentences = overlap_sentences + [sentence]
            current_tokens = encoding.encode(" ".join(current_sentences))

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "text": chunk_text,
            "page": str(page_number),
            "source": PDF_FILE
        })

    return chunks


# =====================================================
# PARSE PDF
# =====================================================

print("\n[1/4] Parsing PDF & Custom Questions...")

all_chunks = []

if os.path.exists(PDF_FILE):
    reader = PdfReader(PDF_FILE)
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        page_chunks = create_chunks(text, page_number)
        all_chunks.extend(page_chunks)
else:
    print(f"Warning: PDF file not found at {PDF_FILE}. Ingesting custom questions only.")


# =====================================================
# INGEST CUSTOM QA DATA
# =====================================================

for qa in CUSTOM_QA_DATA:
    formatted_doc = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
    all_chunks.append({
        "text": formatted_doc,
        "page": qa.get("page", "FAQ"),
        "source": qa.get("source", "Custom FAQ")
    })


# =====================================================
# ADD IDS
# =====================================================

for i, chunk in enumerate(all_chunks):
    chunk["chunk_id"] = i

print("Total chunks (PDF + Custom Questions):", len(all_chunks))


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("\n[2/4] Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)


# =====================================================
# CREATE EMBEDDINGS
# =====================================================

print("\n[3/4] Creating embeddings...")
texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print("Embedding dimension:", len(embeddings[0]))


# =====================================================
# RESET OLD DATABASE
# =====================================================

print("\nRemoving old database...")
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)


# =====================================================
# CREATE CHROMADB & STORE
# =====================================================

print("\n[4/4] Storing in ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

collection.add(
    ids=[str(chunk["chunk_id"]) for chunk in all_chunks],
    documents=[chunk["text"] for chunk in all_chunks],
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "page": chunk["page"],
            "source": chunk["source"]
        }
        for chunk in all_chunks
    ]
)

print("\n================================")
print("       INGESTION COMPLETE")
print("================================")
print("Total Documents in DB:", collection.count())
print("Model:", EMBEDDING_MODEL)
print("Database Path:", DB_PATH)