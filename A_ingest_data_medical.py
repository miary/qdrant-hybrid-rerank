import ollama
import torch
import time
from datasets import load_dataset
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

# --- 1. MODEL AND CLIENT INITIALIZATION ---
print("Initializing models and clients...")

# Qdrant client (persistent on-disk)
# This will create a "./qdrant_db" folder
client = QdrantClient(path="./qdrant_db")

# Ollama models
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"  # Ollama embedding model

# Sparse Model: SPLADE
SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME)
splade_model.eval()

COLLECTION_NAME = "db_hybrid"
BATCH_SIZE = 128  # Upload data in batches

print("Initialization complete.\n")

# --- 2. HELPER FUNCTIONS ---

def get_dense_embedding(text: str) -> list[float]:
    """Generates a dense embedding using Ollama."""
    try:
        return ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)["embedding"]
    except Exception as e:
        print(f"Error getting dense embedding: {e}")
        # Return a zero vector on failure
        return [0.0] * 768 

def get_sparse_vector(text: str) -> models.SparseVector:
    """Generates a SPLADE sparse vector from text."""
    with torch.no_grad():
        inputs = splade_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logits = splade_model(**inputs).logits
        
        vec = torch.log(1 + torch.relu(logits)).squeeze()
        vec_agg = torch.max(vec, dim=0).values
        
        indices = vec_agg.nonzero().squeeze().cpu().tolist()
        values = vec_agg[indices].cpu().tolist()
        
        if not isinstance(indices, list):
            indices = [indices]
            values = [values]
            
        return models.SparseVector(indices=indices, values=values)

# --- 3. DATA INGESTION ---

print(f"Creating/Checking collection '{COLLECTION_NAME}'...")
# Create the collection with both dense and sparse vector configurations
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "vect_dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "vect_sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True,
                )
            )
        }
    )
    print(f"Collection '{COLLECTION_NAME}' created.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")

# Load the dataset from Hugging Face
print("Loading MedicationQA dataset from Hugging Face...")
dataset = load_dataset("truehealth/medicationqa", split="train")
print(f"Dataset loaded. Total entries: {len(dataset)}")

# Process and upsert documents in batches
points = []
for i, doc in enumerate(tqdm(dataset, desc="Processing documents")):
    
    # We will use the 'answer' as the document to be vectorized and stored
    text_to_vectorize = doc['Answer']
    
    # Handle potential empty strings
    if not text_to_vectorize:
        print(f"Skipping empty document at index {i}")
        continue

    # Create the payload
    payload = {
        "question": doc['Question'],
        "answer": doc['Answer'],
        "focus_area": doc['Focus (Drug)']
    }
    
    # Generate vectors
    dense_vec = get_dense_embedding(text_to_vectorize)
    sparse_vec = get_sparse_vector(text_to_vectorize)

    points.append(
        models.PointStruct(
            id=i,
            payload=payload,
            vector={
                "vect_dense": dense_vec,
                "vect_sparse": sparse_vec
            }
        )
    )
    
    # Upsert in batches
    if len(points) >= BATCH_SIZE:
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        points = []

# Upsert any remaining points
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

print(f"\nSuccessfully ingested {len(dataset)} documents into '{COLLECTION_NAME}'.")