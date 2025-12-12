# src/base_retriever.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str], **kwargs) -> np.ndarray:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass

class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers."""
    
    def __init__(self, model_config: dict):
        """
        Initialize dense retriever.
        
        Args:
            model_config: Configuration dictionary from retriever_configs
        """
        from embedding_generator import EmbeddingGenerator
        
        self.model_config = model_config
        self.model_name = model_config["model_name"]
        self.dimension = model_config["dimension"]
        
        logger.info(f"Initializing dense retriever with model: {self.model_name}")
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.model_name
        )
    
    def generate_embeddings(self, texts: List[str], **kwargs) -> np.ndarray:
        """Generate embeddings using the embedding generator."""
        return self.embedding_generator.generate_embeddings(texts, **kwargs)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_generator.get_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name