# configs\experiment_configs.py
"""
Experiment configurations for Elden Ring RAG system evaluation
Integrates with existing retriever configurations
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import existing retriever configurations
from .retriever_configs import RETRIEVER_CONFIGS, get_all_retriever_names, DEFAULT_RETRIEVER

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    name: str
    retriever_name: str
    prompt_type: str
    k: int = 5
    batch_size: int = 8
    use_reranking: bool = False
    use_query_rewriting: bool = False
    description: str = ""

# Prompt configurations
PROMPT_CONFIGS = {
    "vanilla_rag": {
        "name": "Vanilla RAG",
        "description": "Basic RAG prompt with context and fallback"
    },
    "instruction_tuned_rag": {
        "name": "Instruction-Tuned RAG", 
        "description": "Domain-specific prompt for Elden Ring expertise"
    },
    "safety_aware_rag": {
        "name": "Safety-Aware RAG",
        "description": "Enhanced safety with strict context adherence"
    },
    "closed_book_baseline": {
        "name": "Closed-Book Baseline",
        "description": "Simple closed-book prompt without context"
    },
    "closed_book_expert": {
        "name": "Closed-Book Expert",
        "description": "Closed-book with domain expertise framing"
    },
    "hyde_hypothetical": {
        "name": "HyDE Hypothetical",
        "description": "Generate hypothetical answer for query expansion"
    }
}

DEFAULT_PROMPT = "instruction_tuned_rag"

# Evaluation configurations
EVALUATION_CONFIG = {
    "retrieval_metrics": ["recall@1", "recall@3", "recall@5", "mrr"],
    "generation_metrics": ["exact_match", "f1_score", "rouge1", "rouge2", "rougeL"],
    "llm_judge_metrics": ["faithfulness", "relevance", "completeness"],
    "k_values": [1, 3, 5],
    "llm_judge_samples": 30,
    "random_seed": 42
}

# Experiment sets for different comparison types
EXPERIMENT_SETS = {
    # Core comparison: RAG vs Closed-Book
    "rag_vs_closed_book": {
        "name": "RAG vs Closed-Book Comparison",
        "description": "Compare RAG system performance against closed-book baseline",
        "experiments": [
            ExperimentConfig(
                name="rag_baseline",
                retriever_name=DEFAULT_RETRIEVER,
                prompt_type="instruction_tuned_rag",
                description="RAG system with default configuration"
            ),
            ExperimentConfig(
                name="closed_book_baseline", 
                retriever_name="none",
                prompt_type="closed_book_baseline",
                description="Closed-book T5 model without retrieval"
            )
        ]
    },
    
    # Retriever comparison
    "retriever_comparison": {
        "name": "Retriever Comparison",
        "description": "Compare different retrieval methods",
        "experiments": [
            ExperimentConfig(
                name=f"retriever_{retriever}",
                retriever_name=retriever,
                prompt_type=DEFAULT_PROMPT,
                description=f"Using {RETRIEVER_CONFIGS[retriever]['description']}"
            )
            for retriever in get_all_retriever_names()
        ]
    },
    
    # Prompt comparison  
    "prompt_comparison": {
        "name": "Prompt Comparison",
        "description": "Compare different prompt strategies",
        "experiments": [
            ExperimentConfig(
                name=f"prompt_{prompt_type}",
                retriever_name=DEFAULT_RETRIEVER,
                prompt_type=prompt_type,
                description=f"Using {PROMPT_CONFIGS[prompt_type]['name']}"
            )
            for prompt_type in ["vanilla_rag", "instruction_tuned_rag", "safety_aware_rag"]
        ]
    },
    
    # Advanced techniques
    "advanced_techniques": {
        "name": "Advanced Techniques",
        "description": "Test advanced RAG techniques",
        "experiments": [
            ExperimentConfig(
                name="baseline_advanced",
                retriever_name=DEFAULT_RETRIEVER,
                prompt_type=DEFAULT_PROMPT,
                description="Baseline without advanced techniques"
            ),
            ExperimentConfig(
                name="with_reranking",
                retriever_name=DEFAULT_RETRIEVER, 
                prompt_type=DEFAULT_PROMPT,
                use_reranking=True,
                description="Baseline with cross-encoder re-ranking"
            ),
            ExperimentConfig(
                name="with_query_rewriting",
                retriever_name=DEFAULT_RETRIEVER,
                prompt_type=DEFAULT_PROMPT,
                use_query_rewriting=True,
                description="Baseline with HyDE query rewriting"
            )
        ]
    }
}

# Model configurations
MODEL_CONFIGS = {
    "t5-base": {
        "model_name": "t5-base",
        "max_input_length": 512,
        "max_output_length": 256,
        "generation_params": {
            "max_length": 256,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        }
    },
    "t5-small": {
        "model_name": "t5-small", 
        "max_input_length": 512,
        "max_output_length": 256,
        "generation_params": {
            "max_length": 256,
            "num_beams": 4,
            "early_stopping": True
        }
    }
}

DEFAULT_MODEL = "t5-base"

# Path configurations
PATH_CONFIG = {
    "data_dir": "data",
    "processed_dir": "data", 
    "vector_index_dir": "vector_index",
    "evaluation_dir": "evaluation",
    "results_dir": "evaluation/results",
    "logs_dir": "logs"
}

def get_experiment_config(experiment_set: str, experiment_name: str) -> Optional[ExperimentConfig]:
    """
    Get specific experiment configuration
    
    Args:
        experiment_set: Name of the experiment set
        experiment_name: Name of the specific experiment
        
    Returns:
        ExperimentConfig or None if not found
    """
    if experiment_set not in EXPERIMENT_SETS:
        return None
    
    for exp in EXPERIMENT_SETS[experiment_set]["experiments"]:
        if exp.name == experiment_name:
            return exp
    
    return None

def get_all_experiment_sets() -> List[str]:
    """Get list of all available experiment sets"""
    return list(EXPERIMENT_SETS.keys())

def get_prompt_config(prompt_type: str) -> Dict[str, str]:
    """Get configuration for a specific prompt type"""
    return PROMPT_CONFIGS.get(prompt_type, {
        "name": "Unknown",
        "description": "Unknown prompt type"
    })

def get_model_config(model_name: str = None) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_MODEL])

def validate_config() -> List[str]:
    """
    Validate all configurations and return any errors
    
    Returns:
        List of error messages, empty if valid
    """
    errors = []
    
    # Check if all retriever configurations have corresponding indices
    for retriever_name in get_all_retriever_names():
        index_path = os.path.join(
            PATH_CONFIG["vector_index_dir"], 
            f"hollow_knight_{retriever_name}"
        )
        if not os.path.exists(index_path):
            errors.append(f"Vector index not found for retriever '{retriever_name}': {index_path}")
    
    # Check evaluation dataset exists
    eval_dataset_path = os.path.join(PATH_CONFIG["evaluation_dir"], "hollow_knight_qa_set.json")
    if not os.path.exists(eval_dataset_path):
        errors.append(f"Evaluation dataset not found: {eval_dataset_path}")
    
    # Check results directory exists
    results_dir = PATH_CONFIG["results_dir"]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    
    return errors

def get_default_experiment_runner_config() -> Dict[str, Any]:
    """Get default configuration for experiment runner"""
    return {
        "experiment_sets": ["rag_vs_closed_book", "retriever_comparison", "prompt_comparison", "advanced_techniques"],
        "save_results": True,
        "verbose": True,
        "max_workers": 2  # For parallel execution
    }

# Example usage and testing
if __name__ == "__main__":
    print("-> Experiment Configuration Test")
    print("=" * 50)
    
    # Show available experiment sets
    print("Available Experiment Sets:")
    for exp_set in get_all_experiment_sets():
        config = EXPERIMENT_SETS[exp_set]
        print(f"  {exp_set}: {config['name']}")
        print(f"    {config['description']}")
        print(f"    Experiments: {len(config['experiments'])}")
    
    # Validate configuration
    print("\n-> Configuration Validation:")
    errors = validate_config()
    if errors:
        print(" X Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(" V All configurations are valid")
    
    # Show retriever configurations
    print(f"\n-> Available Retrievers: {', '.join(get_all_retriever_names())}")
    for retriever in get_all_retriever_names():
        config = RETRIEVER_CONFIGS[retriever]
        print(f"  {retriever}: {config['description']}")
    
    # Show prompt configurations
    print(f"\n-> Available Prompts:")
    for prompt_type, config in PROMPT_CONFIGS.items():
        print(f"  {prompt_type}: {config['name']} - {config['description']}")