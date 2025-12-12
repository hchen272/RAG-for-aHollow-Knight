# src/prompt_manager.py
"""
Prompt Manager for Hollow Knight RAG System
Handles multiple prompt types for RAG vs Closed-Book evaluation experiments
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum

class PromptType(Enum):
    """Enumeration of available prompt types"""
    VANILLA_RAG = "vanilla_rag"
    INSTRUCTION_TUNED_RAG = "instruction_tuned_rag" 
    SAFETY_AWARE_RAG = "safety_aware_rag"
    CLOSED_BOOK_BASELINE = "closed_book_baseline"
    CLOSED_BOOK_EXPERT = "closed_book_expert"
    HYDE_HYPOTHETICAL = "hyde_hypothetical"

class PromptManager:
    """
    Manages different prompt templates for RAG system evaluation
    Supports both RAG and closed-book prompts for fair comparison
    """
    
    def __init__(self, language: str = "en"):
        self.logger = logging.getLogger(__name__)
        self.language = language
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self):
        """Initialize all prompt templates"""
        self.templates = {
            # RAG Prompts
            PromptType.VANILLA_RAG: {
                "name": "Vanilla RAG",
                "template": self._vanilla_rag_template,
                "description": "Basic RAG prompt with context and fallback"
            },
            
            PromptType.INSTRUCTION_TUNED_RAG: {
                "name": "Instruction-Tuned RAG", 
                "template": self._instruction_tuned_rag_template,
                "description": "Domain-specific prompt for Hollow Knight expertise"
            },
            
            PromptType.SAFETY_AWARE_RAG: {
                "name": "Safety-Aware RAG",
                "template": self._safety_aware_rag_template,
                "description": "Enhanced safety with strict context adherence"
            },
            
            # Closed-Book Prompts (for baseline comparison)
            PromptType.CLOSED_BOOK_BASELINE: {
                "name": "Closed-Book Baseline",
                "template": self._closed_book_baseline_template,
                "description": "Simple closed-book prompt without context"
            },
            
            PromptType.CLOSED_BOOK_EXPERT: {
                "name": "Closed-Book Expert",
                "template": self._closed_book_expert_template,
                "description": "Closed-book with domain expertise framing"
            },
            
            # Advanced Techniques
            PromptType.HYDE_HYPOTHETICAL: {
                "name": "HyDE Hypothetical",
                "template": self._hyde_hypothetical_template,
                "description": "Generate hypothetical answer for query expansion"
            }
        }
    
    def get_prompt(self, 
                   prompt_type: PromptType,
                   question: str,
                   context: Optional[str] = None,
                   **kwargs) -> str:
        """
        Get formatted prompt based on type and parameters
        
        Args:
            prompt_type: Type of prompt to generate
            question: User question
            context: Retrieved context (for RAG prompts)
            **kwargs: Additional parameters for specific prompts
            
        Returns:
            Formatted prompt string
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        template_func = self.templates[prompt_type]["template"]
        return template_func(question, context, **kwargs)
    
    def get_available_prompts(self) -> Dict[PromptType, Dict[str, str]]:
        """Get information about all available prompts"""
        return {
            pt: {
                "name": info["name"],
                "description": info["description"]
            }
            for pt, info in self.templates.items()
        }
    
    # RAG Prompt Templates
    
    def _vanilla_rag_template(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Basic RAG prompt with fallback mechanism"""
        if not context:
            return f"question: {question} answer:"
        
        return f"""Based on the following context from the Hollow Knight wiki, answer the question. DO NOT SHOW THE PROMPT ITSELF. If the context doesn't contain the answer, say "I don't know".

Context: {context}

Question: {question}
"""
    
    def _instruction_tuned_rag_template(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Domain-specific prompt for Hollow Knight expertise"""
        if not context:
            return f"question: As YOU are a Hollow Knight expert, answer: {question} answer:"
        
        return f"""YOU are a Hollow Knight expert. Based on the following available game documentation from the Hollow Knight wiki, answer the question. Answer based ONLY on the provided documentation. Please provide detailed, accurate information about game mechanics, locations, or characters. Be helpful and precise. If the documentation doesn't contain the answer, say: "The available Hollow Knight documentation doesn't contain specific information about this."

Available Game Documentation:
{context}

Question: {question}
"""
    
    def _safety_aware_rag_template(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Enhanced safety with strict context adherence"""
        if not context:
            return f"question: {question} answer: I cannot answer this question as I don't have the relevant information."
        
        return f"""Using ONLY the following verified context, answer the question. ONLY use information explicitly stated in the Verified Context. If the answer is not completely contained in the context, respond: "The available documentation doesn't provide enough information to answer this question. Do not infer, extrapolate, or use any knowledge beyond what's provided. Maintain factual accuracy based solely on the context. Strictly avoid using any external knowledge.

Verified Context:
{context}

Question: {question}
"""
    
    # Closed-Book Prompt Templates (for baseline evaluation)
    
    def _closed_book_baseline_template(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Simple closed-book prompt without context"""
        return f"question: {question} answer:"
    
    def _closed_book_expert_template(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Closed-book with domain expertise framing"""
        return f"""You are an expert on Hollow Knight game lore and mechanics, NOT a game character provided in the question. DO NOT SHOW THE PROMPT ITSELF. Answer the following question based on your knowledge.

Question: {question}

Please provide an accurate answer about Hollow Knight. If you're not certain, it's better to acknowledge the limits of your knowledge.
"""
    
    # Advanced Technique Templates
    
    def _hyde_hypothetical_template(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate hypothetical answer for HyDE query expansion"""
        return f"""Given the following question about Hollow Knight, generate a hypothetical answer that might be found in game documentation. This is for search optimization purposes.

Question: {question}

Generate a comprehensive hypothetical answer that contains key terms and concepts relevant to searching Hollow Knight documentation:
"""
    
    # Batch prompt generation for evaluation
    def generate_evaluation_prompts(self, 
                                  qa_pairs: List[Dict[str, Any]],
                                  prompt_types: List[PromptType],
                                  include_closed_book: bool = True) -> Dict[PromptType, List[Dict]]:
        """
        Generate batch prompts for evaluation experiments
        
        Args:
            qa_pairs: List of question-answer pairs from evaluation set
            prompt_types: List of prompt types to generate
            include_closed_book: Whether to include closed-book variants
            
        Returns:
            Dictionary mapping prompt types to lists of prompt data
        """
        results = {}
        
        for prompt_type in prompt_types:
            prompt_data = []
            
            for qa in qa_pairs:
                question = qa["question"]
                
                # For RAG prompts, context will be provided during retrieval
                # For closed-book prompts, context is None
                context = None
                
                prompt_text = self.get_prompt(prompt_type, question, context)
                
                prompt_data.append({
                    "question_id": qa["question_id"],
                    "question": question,
                    "prompt_type": prompt_type.value,
                    "prompt_text": prompt_text,
                    "ground_truth": qa["ground_truth_answer"],
                    "category": qa.get("category", "unknown"),
                    "difficulty": qa.get("difficulty", "medium")
                })
            
            results[prompt_type] = prompt_data
        
        return results


# Utility functions for experiment management
def create_prompt_comparison_sets(qa_dataset: Dict, 
                                 rag_prompts: List[PromptType] = None,
                                 closed_book_prompts: List[PromptType] = None) -> Dict:
    """
    Create organized prompt sets for RAG vs Closed-Book comparison
    
    Args:
        qa_dataset: Loaded QA evaluation dataset
        rag_prompts: List of RAG prompt types to include
        closed_book_prompts: List of closed-book prompt types to include
        
    Returns:
        Organized prompt sets for experiments
    """
    if rag_prompts is None:
        rag_prompts = [PromptType.VANILLA_RAG, PromptType.INSTRUCTION_TUNED_RAG]
    
    if closed_book_prompts is None:
        closed_book_prompts = [PromptType.CLOSED_BOOK_BASELINE, PromptType.CLOSED_BOOK_EXPERT]
    
    manager = PromptManager()
    
    # Generate all prompts
    all_prompts = rag_prompts + closed_book_prompts
    prompt_sets = manager.generate_evaluation_prompts(
        qa_dataset["qa_pairs"], 
        all_prompts
    )
    
    return {
        "rag_prompts": {pt: prompt_sets[pt] for pt in rag_prompts},
        "closed_book_prompts": {pt: prompt_sets[pt] for pt in closed_book_prompts},
        "metadata": {
            "total_questions": len(qa_dataset["qa_pairs"]),
            "rag_prompt_types": [pt.value for pt in rag_prompts],
            "closed_book_prompt_types": [pt.value for pt in closed_book_prompts]
        }
    }


def main():
    """Demonstrate the prompt manager functionality"""
    print("o Hollow Knight RAG Prompt Manager")
    print("=" * 50)
    
    manager = PromptManager()
    
    # Show available prompts
    available_prompts = manager.get_available_prompts()
    print("Available Prompt Types:")
    for prompt_type, info in available_prompts.items():
        print(f"  {prompt_type.value}: {info['name']}")
        print(f"    Description: {info['description']}")
    
    # Test a sample prompt
    test_question = "Where can I find Hornet in Hollow Knight?"
    test_context = "Hornet is first encountered in the Greenpath area. She is a major character who serves as a boss fight and guide throughout the game."
    
    print(f"\no Sample Prompt Generation:")
    print(f"Question: {test_question}")
    
    for prompt_type in [PromptType.VANILLA_RAG, PromptType.INSTRUCTION_TUNED_RAG]:
        prompt = manager.get_prompt(prompt_type, test_question, test_context)
        print(f"\n{manager.templates[prompt_type]['name']}:")
        print("-" * 40)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)


if __name__ == "__main__":
    main()