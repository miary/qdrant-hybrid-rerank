# -----------------------------------------------------------
# MedicationQA → Ollama API Embeddings → TF-IDF → Qdrant (Hybrid)
# -----------------------------------------------------------

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseVector,
    NamedVector,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import json
from tqdm import tqdm

# -----------------------------------------------------------
# 1. Load MedicationQA dataset
# -----------------------------------------------------------
print("📚 Loading MedicationQA dataset...")
dataset = load_dataset("truehealth/medicationqa", split="train")
dataset = dataset.select(range(600))

texts = [f"Q: {x['Question']} A: {x['Answer']}" for x in dataset]

# -----------------------------------------------------------
# 2. Define Ollama API embedding helper
# -----------------------------------------------------------
def get_ollama_embedding(text, model="nomic-embed-text"):
    """
    Get an embedding via Ollama's HTTP API.
    Make sure Ollama is running: ollama serve
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if "embedding" in data:
            return data["embedding"]
        elif "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
        else:
            print("⚠️ Unexpected Ollama API response:", data)
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API request failed: {e}")
        return None

# -----------------------------------------------------------
# 3. Generate dense embeddings
# -----------------------------------------------------------
print("🧠 Generating dense embeddings via Ollama API...")

dense_vectors = []
valid_items = []

for i, t in enumerate(tqdm(texts, desc="Embedding texts", ncols=80, unit="item")):
    emb = get_ollama_embedding(t)
    if emb is not None:
        dense_vectors.append(emb)
        valid_items.append(dataset[i])
    else:
        tqdm.write(f"⚠️ Skipping item {i}: embedding failed")

if not dense_vectors:
    raise RuntimeError("❌ No valid embeddings returned from Ollama API!")

dense_dim = len(dense_vectors[0])
print(f"✅ Got {len(dense_vectors)} valid embeddings (dim={dense_dim})")

# -----------------------------------------------------------
# 4. Build sparse (TF-IDF) vectors
# -----------------------------------------------------------
print("🧩 Building sparse TF-IDF features...")
texts_valid = [f"Q: {x['Question']} A: {x['Answer']}" for x in valid_items]

tfidf = TfidfVectorizer(max_features=10000)
tfidf.fit(texts_valid)
tfidf_matrix = tfidf.transform(texts_valid)

def to_sparse_vector(row):
    coo = row.tocoo()
    return SparseVector(indices=coo.col.tolist(), values=coo.data.tolist())

sparse_vectors = [to_sparse_vector(tfidf_matrix[i]) for i in range(tfidf_matrix.shape[0])]
print(f"✅ Created {len(sparse_vectors)} sparse vectors.")

# -----------------------------------------------------------
# 5. Connect to Qdrant
# -----------------------------------------------------------
client = QdrantClient("http://localhost:6333")
collection_name = "medicationqa_collection"

# -----------------------------------------------------------
# 6. Create collection if not exists
# -----------------------------------------------------------
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "vect_dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "vect_sparse": SparseVectorParams()
        },
    )
    print(f"✅ Created collection: {collection_name}")
else:
    print(f"ℹ️ Collection '{collection_name}' already exists.")

# -----------------------------------------------------------
# 7. Insert data
# -----------------------------------------------------------
print("⬆️ Inserting data into Qdrant...")

points = [
    PointStruct(
        id=i,
        vector={
            "vect_dense": dense_vectors[i],
            "vect_sparse": sparse_vectors[i],
        },
        payload={
            "question": valid_items[i]["Question"],
            "answer": valid_items[i]["Answer"],
        },
    )
    for i in range(len(valid_items))
]

client.upsert(collection_name=collection_name, points=points)
print(f"✅ Inserted {len(points)} points into Qdrant collection '{collection_name}'.")

# -----------------------------------------------------------
# 8. Hybrid Fusion Search
# -----------------------------------------------------------
query_text = "What medication helps lower blood pressure?"
print(f"\n🔍 Running hybrid search for query: '{query_text}'")

# Dense query embedding via Ollama API
query_dense = get_ollama_embedding(query_text)

# Sparse query vector via TF-IDF
query_sparse_matrix = tfidf.transform([query_text])
query_sparse = to_sparse_vector(query_sparse_matrix[0])

# Perform fusion search (dense + sparse)
results = client.query_points(
    collection_name=collection_name,
    query=[
        NamedVector(name="vect_dense", vector=query_dense),
        NamedVector(name="vect_sparse", vector=query_sparse),
    ],
    limit=5,
    fusion="reciprocal_rank_fusion",
)

# -----------------------------------------------------------
# 9. Display results
# -----------------------------------------------------------
for i, point in enumerate(results.points):
    print(f"\nRank {i+1}")
    print(f"Score: {point.score:.4f}")
    print(f"Question: {point.payload['question']}")
    print(f"Answer: {point.payload['answer']}")
