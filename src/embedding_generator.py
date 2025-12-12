# src/embedding_generator.py

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedding generator with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on (None for auto-detection)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          batch_size: int = 32, 
                          show_progress_bar: bool = True) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def __call__(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Make the class callable for convenience."""
        return self.generate_embeddings(texts, **kwargs)