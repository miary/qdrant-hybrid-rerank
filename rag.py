import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import CrossEncoder
import uuid
from typing import List, Dict, Tuple
import numpy as np

OLLAMA_EMBED_MODEL = "qwen3-embedding:4b" 
OLLAMA_LLM_MODEL = "llama3.2" #"qwen3:14b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class RAGSystem:
    def __init__(
        self,
        collection_name: str = "my_documents",
        embedding_model: str = OLLAMA_EMBED_MODEL,
        llm_model: str = OLLAMA_LLM_MODEL,
        reranker_model: str = RERANKER_MODEL,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        """
        Initialize the RAG system with Ollama and Qdrant
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Ollama embedding model to use
            llm_model: Ollama LLM model for generation
            reranker_model: HuggingFace reranker model
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize reranker
        self.reranker = CrossEncoder(reranker_model)
        
        # Get embedding dimension
        sample_embedding = self._get_embedding("sample text")
        self.embedding_dim = len(sample_embedding)
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Ollama
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response['embedding']
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """
        Add documents to the vector database
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dictionaries for each document
        """
        if metadata is None:
            metadata = [{} for _ in documents]
        
        points = []
        for idx, (doc, meta) in enumerate(zip(documents, metadata)):
            embedding = self._get_embedding(doc)
            point_id = str(uuid.uuid4())
            
            payload = {
                "text": doc,
                **meta
            }
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(documents)} documents")
        
        # Upload points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(documents)} documents to the collection")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant documents with reranking
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve initially
            rerank_top_k: Number of documents to return after reranking
            
        Returns:
            List of reranked documents with scores
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        ).points
        
        if not search_results:
            return []
        
        # Prepare documents for reranking
        documents = [hit.payload['text'] for hit in search_results]
        
        # Rerank using cross-encoder
        rerank_scores = self.reranker.predict([
            (query, doc) for doc in documents
        ])
        
        # Combine results with rerank scores
        reranked_results = []
        for hit, rerank_score in zip(search_results, rerank_scores):
            reranked_results.append({
                'text': hit.payload['text'],
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'},
                'original_score': hit.score,
                'rerank_score': float(rerank_score),
                'id': hit.id
            })
        
        # Sort by rerank score and return top_k
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_results[:rerank_top_k]
    
    def generate_response(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 3,
        system_prompt: str = None
    ) -> Dict:
        """
        Generate response using RAG
        
        Args:
            query: User query
            top_k: Number of documents to retrieve initially
            rerank_top_k: Number of documents to use after reranking
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary containing response and source documents
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k, rerank_top_k)
        
        if not relevant_docs:
            return {
                'response': "I couldn't find any relevant information to answer your question.",
                'sources': []
            }
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Prepare prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context. 
If the context doesn't contain enough information to answer the question, say so clearly."""
        
        prompt = f"""Context:
{context}

Question: {query}

Answer based on the context provided above:"""
        
        # Generate response using Ollama
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        return {
            'response': response['message']['content'],
            'sources': relevant_docs,
            'context': context
        }
    
    def delete_collection(self):
        """Delete the collection from Qdrant"""
        self.qdrant_client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


# Utility function for chunking large documents
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


# Example usage
def main():
    # Initialize RAG system
    rag = RAGSystem(
        collection_name="my_knowledge_base",
        embedding_model=OLLAMA_EMBED_MODEL,
        llm_model=OLLAMA_LLM_MODEL
    )
    
    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation to produce more accurate and informed responses.",
        "Vector databases store data as high-dimensional vectors, enabling efficient similarity search and retrieval operations.",
        "Qdrant is an open-source vector database that provides fast and scalable similarity search capabilities.",
        "Ollama is a tool that allows you to run large language models locally on your machine.",
        "Embeddings are numerical representations of text that capture semantic meaning in a vector space.",
        "Reranking improves search results by re-scoring initially retrieved documents using more sophisticated models.",
    ]
    
    # Add metadata (optional)
    metadata = [
        {"source": "python_docs", "category": "programming"},
        {"source": "ml_guide", "category": "ai"},
        {"source": "rag_paper", "category": "ai"},
        {"source": "db_guide", "category": "database"},
        {"source": "qdrant_docs", "category": "database"},
        {"source": "ollama_docs", "category": "tools"},
        {"source": "nlp_book", "category": "ai"},
        {"source": "ir_paper", "category": "ai"},
    ]
    
    # Add documents to the system
    print("Adding documents to RAG system...")
    rag.add_documents(documents, metadata)
    
    # Query the system
    print("\n" + "="*50)
    query = "What is RAG and how does it work?"
    print(f"Query: {query}")
    print("="*50 + "\n")
    
    result = rag.generate_response(query, top_k=5, rerank_top_k=2)
    
    print("Response:")
    print(result['response'])
    print("\n" + "="*50)
    print("Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. (Rerank Score: {source['rerank_score']:.4f})")
        print(f"   Text: {source['text'][:100]}...")
        print(f"   Metadata: {source['metadata']}")


if __name__ == "__main__":
    main()