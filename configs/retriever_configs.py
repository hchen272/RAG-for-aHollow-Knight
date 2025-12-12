# src/configs/retriever_configs.py

RETRIEVER_CONFIGS = {
    "miniLM": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "similarity": "cosine",
        "description": "Lightweight model, good balance of speed and quality, dense"
    },
    "mpnet": {
        "model_name": "all-mpnet-base-v2",
        "dimension": 768, 
        "similarity": "cosine",
        "description": "Higher quality model, better for complex queries, dense"
    },
    "bge_small": {
        "model_name": "BAAI/bge-small-en",
        "dimension": 384,
        "similarity": "cosine",
        "description": "Specialized for retrieval tasks, dense"
    }
}

DEFAULT_RETRIEVER = "miniLM"

def get_retriever_config(retriever_name: str = None) -> dict:
    """Get retriever configuration by name."""
    if retriever_name is None:
        retriever_name = DEFAULT_RETRIEVER
    
    if retriever_name not in RETRIEVER_CONFIGS:
        raise ValueError(f"Retriever '{retriever_name}' not found. Available: {list(RETRIEVER_CONFIGS.keys())}")
    
    return RETRIEVER_CONFIGS[retriever_name].copy()

def get_all_retriever_names() -> list:
    """Get list of all available retriever names."""
    return list(RETRIEVER_CONFIGS.keys())