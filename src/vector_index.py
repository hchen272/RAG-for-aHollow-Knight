# src/vector_index.py

import faiss
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorIndex:
    """
    FAISS-based vector index for Hollow Knight RAG system
    Handles creation, storage, and retrieval of vector embeddings
    """
    
    def __init__(self, index_path: str, dimension: int = 384):
        """
        Initialize vector index manager
        
        Args:
            index_path: Directory path for storing index files
            dimension: Dimension of embedding vectors
        """
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
    def create_index(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """
        Create FAISS index from embeddings and documents
        
        Args:
            embeddings: Numpy array of embeddings (n_vectors x dimension)
            documents: List of document metadata corresponding to embeddings
        """
        if len(embeddings) != len(documents):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of documents ({len(documents)})")
        
        logger.info(f"Creating FAISS index with {len(embeddings)} vectors of dimension {self.dimension}")
        
        # Create IndexFlatIP for cosine similarity (vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = documents
        
        logger.info(f"Created index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Search for similar vectors in the index
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, results)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty or not initialized")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    **self.metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return distances[0], indices[0], results
    
    def save(self) -> None:
        """Save index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Create directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = self.index_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save configuration
        config = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "index_type": "FlatIP"
        }
        
        config_file = self.index_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved index to {self.index_path} with {self.index.ntotal} vectors")
    
    def load(self) -> None:
        """Load index and metadata from disk"""
        # Load FAISS index
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        # Load metadata
        metadata_file = self.index_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load configuration
        config_file = self.index_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.dimension = config.get("dimension", 384)
        
        logger.info(f"Loaded index from {self.index_path} with {self.index.ntotal} vectors")
    
    def exists(self) -> bool:
        """Check if index files exist"""
        required_files = ["index.faiss", "metadata.json", "config.json"]
        return all((self.index_path / file).exists() for file in required_files)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index is None:
            return {"total_vectors": 0, "dimension": self.dimension}
        
        # Count documents by type
        type_counts = {}
        for doc in self.metadata:
            doc_type = doc.get('doc_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "document_types": type_counts
        }


class HollowKnightVectorIndexBuilder:
    """
    Builder class for creating Hollow Knight vector indices
    Handles the complete pipeline from documents to indexed vectors
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en"):
        """
        Initialize the vector index builder
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
        """
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
    def load_documents(self, data_path: str) -> List[Dict[str, Any]]:
        """Load formatted documents from JSON file"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            self.logger.info(f"Loaded {len(documents)} documents from {data_path}")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            raise
    
    def prepare_documents(self, documents: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Prepare documents for embedding generation
        
        Returns:
            Tuple of (texts for embedding, document metadata)
        """
        texts = []
        metadata_list = []
        
        for doc in documents:
            # Create retrieval text using new JSON format
            title = doc.get('title', 'Unknown')
            doc_type = doc.get('doc_type', 'General')
            content = doc.get('content', '')
            
            retrieval_text = f"{title} - {doc_type}. {content}"
            texts.append(retrieval_text)
            
            # Prepare metadata with new JSON fields
            metadata = {
                "chunk_id": doc.get("chunk_id", ""),
                "title": title,
                "doc_type": doc_type,
                "content": content,
                "url": doc.get("url", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "total_chunks": doc.get("total_chunks", 0),
                "content_length": doc.get("content_length", 0)
            }
            metadata_list.append(metadata)
        
        return texts, metadata_list
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model
            model = SentenceTransformer(self.embedding_model)
            
            # Generate embeddings with progress
            self.logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except ImportError:
            self.logger.error("sentence-transformers not installed. Please install: pip install sentence-transformers")
            raise
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def build_index(self, data_path: str, output_path: str, dimension: int = 384) -> VectorIndex:
        """
        Build complete vector index from documents
        
        Args:
            data_path: Path to formatted documents JSON
            output_path: Directory to save the index
            dimension: Expected embedding dimension
            
        Returns:
            Initialized VectorIndex instance
        """
        # Load and prepare documents
        documents = self.load_documents(data_path)
        texts, metadata = self.prepare_documents(documents)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Verify dimension
        if embeddings.shape[1] != dimension:
            self.logger.warning(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {dimension}")
            dimension = embeddings.shape[1]
        
        # Create and save index
        vector_index = VectorIndex(output_path, dimension)
        vector_index.create_index(embeddings, metadata)
        vector_index.save()
        
        # Print statistics
        stats = vector_index.get_stats()
        print(f"Vector index built successfully!")
        print(f"Location: {output_path}")
        print(f"Statistics:")
        print(f"  - Total vectors: {stats['total_vectors']}")
        print(f"  - Dimension: {stats['dimension']}")
        print(f"  - Document types:")
        for doc_type, count in stats['document_types'].items():
            print(f"    - {doc_type}: {count}")
        
        return vector_index


def build_hollow_knight_index(
    data_path: str = "../data/hollow_knight_fandom_rag_optimized.json",
    output_dir: str = "../vector_index/hollow_knight",
    model_name: str = "BAAI/bge-small-en"
) -> VectorIndex:
    """
    Convenience function to build Hollow Knight vector index
    
    Args:
        data_path: Path to formatted data JSON
        output_dir: Directory to save index
        model_name: Embedding model name
        
    Returns:
        VectorIndex instance
    """
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print("Building Hollow Knight Vector Index")
    print("=" * 50)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_name}")
    print()
    
    # Build index
    builder = HollowKnightVectorIndexBuilder(embedding_model=model_name)
    vector_index = builder.build_index(data_path, output_dir)
    
    return vector_index


def main():
    """Main function for building vector index"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Hollow Knight vector index")
    parser.add_argument("--data", default="data/hollow_knight_fandom_rag_optimized.json", 
                       help="Path to formatted data JSON")
    parser.add_argument("--output", default="vector_index/hollow_knight",
                       help="Output directory for index")
    parser.add_argument("--model", default="BAAI/bge-small-en",
                       help="Sentence transformer model name")
    
    args = parser.parse_args()
    
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Build index
        vector_index = build_hollow_knight_index(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.model
        )
        
        # Test the index
        print(f"Testing index...")
        test_query = "How to defeat Absolute Radiance"
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.model)
            query_embedding = model.encode([test_query], normalize_embeddings=True)[0]
            
            distances, indices, results = vector_index.search(query_embedding, k=3)
            
            print(f"Query: '{test_query}'")
            print(f"Top result: {results[0]['title']} (score: {results[0]['score']:.3f})")
            print("Index test passed!")
            
        except Exception as e:
            print(f"Index test failed: {e}")
        
    except Exception as e:
        print(f"Failed to build index: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()