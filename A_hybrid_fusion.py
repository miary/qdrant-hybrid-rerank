from qdrant_client.models import SparseVector, Prefetch
from qdrant_client import QdrantClient
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import json


client = QdrantClient("http://localhost:6333")
collection_name = "medicationqa_collection"


def to_sparse_vector(row):
    coo = row.tocoo()
    return SparseVector(indices=coo.col.tolist(), values=coo.data.tolist())


# -----------------------------------------------------------
# Ollama API Helper Functions
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


def generate_ollama_response(prompt, model="llama3.2", stream=True):
    """
    Generate a response using Ollama's chat API.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream
            },
            stream=stream,
            timeout=120
        )
        response.raise_for_status()
        
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    if data.get("done", False):
                        break
            print()  # New line after streaming
            return full_response
        else:
            data = response.json()
            return data.get("response", "")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama generation failed: {e}")
        return None


def hybrid_search(query_text, limit=5):
    """
    Perform hybrid search combining dense and sparse vectors.
    Returns the top results from Qdrant.
    """
    print("\n⏳ Step 1: Generating dense embedding via Ollama...")
    dense_start = time.time()
    query_dense = get_ollama_embedding(query_text)
    dense_time = time.time() - dense_start
    
    if query_dense is None:
        print("❌ Failed to get embedding from Ollama. Make sure Ollama is running.")
        return None, {}
    
    print(f"✅ Dense embedding generated (dimension: {len(query_dense)}) in {dense_time:.3f}s")
    
    print("\n⏳ Step 2: Generating sparse embedding via TF-IDF...")
    sparse_start = time.time()
    tfidf = TfidfVectorizer(max_features=10000)
    tfidf.fit([query_text])
    
    query_sparse_matrix = tfidf.transform([query_text])
    query_sparse = to_sparse_vector(query_sparse_matrix[0])
    sparse_time = time.time() - sparse_start
    print(f"✅ Sparse embedding generated ({len(query_sparse.indices)} non-zero features) in {sparse_time:.3f}s")
    
    print("\n⏳ Step 3: Executing hybrid search with prefetch...")
    print(f"   - Prefetching top 20 from dense vector (vect_dense)")
    print(f"   - Prefetching top 20 from sparse vector (vect_sparse)")
    print(f"   - Fusing results and returning top {limit}")
    
    search_start = time.time()
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=query_dense, using="vect_dense", limit=20),
            Prefetch(query=query_sparse, using="vect_sparse", limit=20),
        ],
        query=query_dense,
        using="vect_dense",
        limit=limit,
    )
    search_time = time.time() - search_start
    print(f"✅ Search completed in {search_time:.3f}s")
    
    timing_info = {
        "dense_time": dense_time,
        "sparse_time": sparse_time,
        "search_time": search_time,
        "total_retrieval_time": dense_time + sparse_time + search_time
    }
    
    return results, timing_info


def build_rag_prompt(query, retrieved_docs):
    """
    Build a RAG prompt with the query and retrieved context.
    """
    context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        context += f"\n[Document {i}]\n"
        context += f"Question: {doc.payload['question']}\n"
        context += f"Answer: {doc.payload['answer']}\n"
    
    prompt = f"""You are a helpful medical assistant. Use the following retrieved documents to answer the user's question. If the documents do not contain relevant information, say so. If the documents are empty, say that You do not know. Your response should be uniquely based on the document content and nothing else. Do not mention the document or source. Do not provide general information. Do not offer any suggestions.

Retrieved Context:
{context}

User Question: {query}

Answer: Provide a clear, accurate, and helpful response based on the context above."""
    
    return prompt


def rag_pipeline(query_text, model="llama3.2", top_k=5):
    """
    Complete RAG pipeline: Retrieve relevant documents and generate response.
    """
    print("=" * 70)
    print("🚀 Starting RAG Pipeline with Hybrid Fusion Search")
    print("=" * 70)
    print(f"\n📝 Query: {query_text}")
    
    start_time = time.time()
    
    # Step 1: Retrieve relevant documents
    print("\n" + "=" * 70)
    print("📚 RETRIEVAL PHASE")
    print("=" * 70)
    
    results, timing_info = hybrid_search(query_text, limit=top_k)
    
    if results is None:
        return
    
    # Display retrieved documents
    print("\n" + "=" * 70)
    print("📊 RETRIEVED DOCUMENTS")
    print("=" * 70)
    print(f"📈 Total results: {len(results.points)}")
    print("-" * 70)
    
    for i, point in enumerate(results.points):
        print(f"\n🏆 Rank {i+1}")
        print(f"   Score: {point.score:.4f}")
        print(f"   Question: {point.payload['question']}")
        print(f"   Answer: {point.payload['answer'][:100]}...")
    
    # Step 2: Build prompt with context
    print("\n" + "=" * 70)
    print("🔨 BUILDING RAG PROMPT")
    print("=" * 70)
    
    prompt = build_rag_prompt(query_text, results.points)
    print(f"✅ Prompt built with {len(results.points)} documents")
    
    # Step 3: Generate response
    print("\n" + "=" * 70)
    print("🤖 GENERATION PHASE")
    print("=" * 70)
    print(f"⏳ Generating response using {model}...\n")
    
    generation_start = time.time()
    response = generate_ollama_response(prompt, model=model, stream=True)
    generation_time = time.time() - generation_start
    
    if response is None:
        print("❌ Failed to generate response")
        return
    
    # Display timing information
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("⏱️  EXECUTION TIME BREAKDOWN")
    print("=" * 70)
    print(f"Dense embedding:  {timing_info['dense_time']:.3f}s ({timing_info['dense_time']/total_time*100:.1f}%)")
    print(f"Sparse embedding: {timing_info['sparse_time']:.3f}s ({timing_info['sparse_time']/total_time*100:.1f}%)")
    print(f"Hybrid search:    {timing_info['search_time']:.3f}s ({timing_info['search_time']/total_time*100:.1f}%)")
    print(f"LLM generation:   {generation_time:.3f}s ({generation_time/total_time*100:.1f}%)")
    print(f"{'─' * 70}")
    print(f"TOTAL TIME:       {total_time:.3f}s")
    print("=" * 70)
    
    return {
        "query": query_text,
        "retrieved_docs": results.points,
        "response": response,
        "timing": {
            **timing_info,
            "generation_time": generation_time,
            "total_time": total_time
        }
    }


# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
if __name__ == "__main__":
    # Example query
    query = "What are the side effects of aspirin?"
    query = "What medication helps with high cholesterol?"
    query = "Can I drink tea with azithromycin?"
    query = "Can I take lansoprazole while pregnant?"
    query = "What are the common uses of metformin?"
    query = "Is metformin FDA approved for weight loss?"
    query = "can you take how to combine dapaliflozin with metformin?"
    query = "What medication helps lower blood pressure?"
    query = "Can you have cocktail with tylenol?"
    # Run RAG pipeline
    result = rag_pipeline(
        query_text=query,
        model="llama3.2",  # Change to your preferred Ollama model
        top_k=3  # Number of documents to retrieve
    )
    
    # You can also run multiple queries
    print("\n\n" + "🔄" * 35)
    print("\n")
    