# src/hollow_knight_rag_system.py

import json
import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional, Any
import re
from vector_index import VectorIndex
from prompt_manager import PromptManager, PromptType
import os
import numpy as np
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the retriever configuration system
from configs.retriever_configs import get_retriever_config, get_all_retriever_names, DEFAULT_RETRIEVER

# Only show critical errors
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)

class HollowKnightRAGSystem:
    """
    RAG system for Hollow Knight game Q&A
    Provides accurate information about bosses, enemies, NPCs, and game mechanics
    """
    
    def __init__(self, 
                 base_vector_index_dir: str = "vector_index",
                 retriever_name: str = None,
                 model_name: str = "t5-base",
                 device: str = None,
                 prompt_type: str = "instruction_tuned_rag"):
        """
        Initialize the Hollow Knight RAG system with multi-retriever support
        
        Args:
            base_vector_index_dir: Base directory containing vector indices
            retriever_name: Name of the retriever to use
            model_name: T5 model name for generation
            device: Device to run models on
            prompt_type: Type of prompt to use
        """
        self.logger = logging.getLogger(__name__)
        self.base_vector_index_dir = base_vector_index_dir
        
        # Set up retriever
        if retriever_name is None:
            retriever_name = DEFAULT_RETRIEVER
        
        self.retriever_name = retriever_name
        self.retriever_config = get_retriever_config(retriever_name)
        
        # Build vector index path
        self.vector_index_path = os.path.join(
            base_vector_index_dir, 
            f"hollow_knight_{retriever_name}"
        )
        
        # Initialize vector index
        self.vector_index = VectorIndex(self.vector_index_path)
        self.vector_index.load()
        
        # Initialize embedding generator with the correct model
        from embedding_generator import EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.retriever_config["model_name"],
            device=device
        )
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        try:
            self.prompt_type = PromptType(prompt_type)
        except ValueError:
            print(f"Warning: Unknown prompt type '{prompt_type}', using default")
            self.prompt_type = PromptType.INSTRUCTION_TUNED_RAG

        # Initialize T5 model and tokenizer
        try:
            self.model_name = model_name
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Set device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            self.model.to(self.device)
            
            print(f" V Hollow Knight RAG System ready")
            print(f"   Retriever: {retriever_name}")
            print(f"   Generator: {model_name}")
            print(f"   Prompt Type: {prompt_type}")
            
        except Exception as e:
            print(f" X Failed to initialize: {e}")
            raise
    
    def switch_retriever(self, retriever_name: str) -> bool:
        """
        Switch to a different retriever at runtime
        
        Args:
            retriever_name: Name of the retriever to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new retriever config
            new_config = get_retriever_config(retriever_name)
            
            # Build new vector index path
            new_vector_index_path = os.path.join(
                self.base_vector_index_dir, 
                f"hollow_knight_{retriever_name}"
            )
            
            # Check if the index exists
            if not os.path.exists(new_vector_index_path):
                self.logger.error(f"Vector index not found for retriever {retriever_name}: {new_vector_index_path}")
                return False
            
            # Update components
            self.retriever_name = retriever_name
            self.retriever_config = new_config
            self.vector_index_path = new_vector_index_path
            
            # Reload vector index
            self.vector_index = VectorIndex(self.vector_index_path)
            self.vector_index.load()
            
            # Update embedding model
            from embedding_generator import EmbeddingGenerator
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.retriever_config["model_name"],
                device=self.device
            )
            
            self.logger.info(f"Switched to retriever: {retriever_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch retriever to {retriever_name}: {e}")
            return False
    
    def get_available_retrievers(self) -> List[str]:
        """
        Get list of available retriever names
        
        Returns:
            List of available retriever names
        """
        return get_all_retriever_names()
    
    def get_current_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the current retriever
        
        Returns:
            Dictionary with retriever information
        """
        return {
            "name": self.retriever_name,
            "model": self.retriever_config["model_name"],
            "dimension": self.retriever_config["dimension"],
            "description": self.retriever_config.get("description", ""),
            "index_path": self.vector_index_path
        }
    
    def get_available_prompt_types(self) -> List[str]:
        """
        Get list of available prompt types
        
        Returns:
            List of available prompt type names
        """
        available_prompts = self.prompt_manager.get_available_prompts()
        return [pt.value for pt in available_prompts.keys()]
    
    def switch_prompt(self, prompt_type: str) -> bool:
        """
        Switch to a different prompt type
        
        Args:
            prompt_type: Name of the prompt type to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            new_prompt_type = PromptType(prompt_type)
            self.prompt_type = new_prompt_type
            self.logger.info(f"Switched to prompt type: {prompt_type}")
            return True
        except ValueError:
            self.logger.error(f"Unknown prompt type: {prompt_type}")
            return False

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for the query
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embeddings(
                [query], 
                batch_size=1,
                show_progress_bar=False
            )[0]

            # Search using the vector index
            distances, indices, results = self.vector_index.search(query_embedding, k=k*2)
            
            # Filter and format results
            retrieved_docs = []
            for i, (distance, result) in enumerate(zip(distances, results)):
                content = result.get("content", "")
                title = result.get("title", "")
                retrieved_docs.append({
                    'content': result.get('content', ''),
                    'title': result.get('title', ''),
                    'type': result.get('type', ''),
                    'score': float(distance),
                })
            
            return sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)[:k], sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)[:k]
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []

    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No specific information found in the Hollow Knight game data."
        
        context_parts = []
        for doc in retrieved_docs:
            # Clean and format content
            content = doc['content']

            content = re.sub(r'\s+', ' ', content).strip()
            
            if content and len(content) > 30:
                # Truncate very long content while preserving sentences
                if len(content) > 400:
                    if '.' in content[:400]:
                        cutoff = content[:400].rfind('.') + 1
                        content = content[:cutoff]
                    else:
                        content = content[:397] + "..."
                
                context_parts.append(f"{content}")
        
        return "\n\n".join(context_parts) if context_parts else "Limited information available from the game data."

    def build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt using the prompt manager
        
        Args:
            question: User's question
            context: Formatted context from retrieved documents
            
        Returns:
            Formatted prompt for the language model
        """
        return self.prompt_manager.get_prompt(self.prompt_type, question, context)

    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using T5 model
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated answer
        """
        try:
            # Tokenize with context
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=4096,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=1000,  # Increased for longer answers
                    num_beams=6,     # More beams for better quality
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    length_penalty=1.5,  # Encourage longer answers
                    do_sample=False,
                    min_length=30,   # Minimum answer length
                )
            
            # Decode and clean up answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt fragments if present
            answer = re.sub(r'^\s*(question:|context:|answer:)\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Ensure proper formatting
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
                
            return answer
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return "I need to consult the Hollow Knight game data for more information about this."

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Main query method - retrieve information and generate answer
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with query results
        """
        # Retrieve relevant documents
        all_docs, retrieved_docs = self.retrieve(question, k)
        
        # Format context
        context = self.format_context(retrieved_docs)
        all_docs_formatted = self.format_context(all_docs)

        # Build prompt
        prompt = self.build_prompt(question, context)
        
        # Generate answer
        answer = self.generate_answer(prompt)
        
        # Calculate confidence based on retrieval scores
        confidence = 0.0
        if retrieved_docs:
            avg_score = np.mean([doc["score"] for doc in retrieved_docs])
            confidence = min(avg_score, 1.0)
        
        return {
            "question": question,
            "answer": answer,
            "all_docs_formatted": context,
            "confidence": float(confidence),
            "sources_count": len(retrieved_docs),
            "retriever_used": self.retriever_name,
            "prompt_type": self.prompt_type.value,
            "retrieved_docs": [
                {
                    "title": doc["title"],
                    "type": doc["type"],
                    "score": doc["score"]
                }
                for doc in retrieved_docs
            ]
        }
    
    def generate_evaluation_prompts(self, 
                                  qa_pairs: List[Dict[str, Any]],
                                  prompt_types: List[PromptType] = None) -> Dict[PromptType, List[Dict]]:
        """
        Generate batch prompts for evaluation experiments
        
        Args:
            qa_pairs: List of question-answer pairs from evaluation set
            prompt_types: List of prompt types to generate
            
        Returns:
            Dictionary mapping prompt types to lists of prompt data
        """
        if prompt_types is None:
            prompt_types = [
                PromptType.VANILLA_RAG,
                PromptType.INSTRUCTION_TUNED_RAG,
                PromptType.SAFETY_AWARE_RAG,
                PromptType.CLOSED_BOOK_BASELINE,
                PromptType.CLOSED_BOOK_EXPERT
            ]
        
        return self.prompt_manager.generate_evaluation_prompts(qa_pairs, prompt_types)


def main():
    """Demonstration of Hollow Knight RAG system with multi-retriever support"""
    print("🎮 Hollow Knight Game Assistant - RAG System")
    print("=" * 60)
    
    try:
        # Get available retrievers
        available_retrievers = get_all_retriever_names()
        print(f"Available retrievers: {', '.join(available_retrievers)}")
        
        # Initialize system
        rag_system = HollowKnightRAGSystem(
            base_vector_index_dir="vector_index",
            retriever_name=available_retrievers[0] if available_retrievers else DEFAULT_RETRIEVER,
            model_name="google/flan-t5-base",
            prompt_type="instruction_tuned_rag"
        )
        
        # Show available prompt types
        available_prompts = rag_system.get_available_prompt_types()
        print(f"Available prompt types: {', '.join(available_prompts)}")
        
        # Test with different prompt types
        test_questions = [
            "Where is False Knight?",
            "Where can I find the Mantis Claw?",
            "What does Salubra sell?",
        ]
        
        # Test each prompt type briefly
        for prompt_type in available_prompts[:3]:  # Test first 3 prompt types
            print(f"\n Testing with prompt type: {prompt_type}")
            print("-" * 40)
            
            if rag_system.switch_prompt(prompt_type):
                retriever_info = rag_system.get_current_retriever_info()
                print(f"Retriever: {retriever_info['name']}")
                
                # Test one question per prompt type for demo
                if test_questions:
                    response = rag_system.query(test_questions[0])
                    
                    print(f"\nQ: {test_questions[0]}")
                    print(f"A: {response['answer'][:150]}...")
                    print(f"Confidence: {response['confidence']:.3f}, Sources: {response['sources_count']}")
        
        # Test retriever switching
        if len(available_retrievers) > 1:
            print(f"\n Testing retriever switching...")
            next_retriever = available_retrievers[1]
            if rag_system.switch_retriever(next_retriever):
                switched_response = rag_system.query("What is the Hollow Knight?")
                print(f"After switching to {next_retriever}:")
                print(f"Q: What is the Hollow Knight?")
                print(f"A: {switched_response['answer'][:100]}...")
                print(f"Confidence: {switched_response['confidence']:.3f}")
        
        # Demonstrate batch prompt generation for evaluation
        print(f"\n Testing batch prompt generation...")
        sample_qa_pairs = [
            {
                "question_id": 1,
                "question": "How do I beat the Radiance?",
                "ground_truth_answer": "The Radiance is the true final boss...",
                "category": "boss",
                "difficulty": "hard"
            },
            {
                "question_id": 2,
                "question": "Where is the City of Tears?",
                "ground_truth_answer": "The City of Tears is located in the central part of Hallownest...",
                "category": "location",
                "difficulty": "medium"
            }
        ]
        
        eval_prompts = rag_system.generate_evaluation_prompts(
            sample_qa_pairs,
            [PromptType.VANILLA_RAG, PromptType.INSTRUCTION_TUNED_RAG]
        )
        
        print(f"Generated {sum(len(prompts) for prompts in eval_prompts.values())} evaluation prompts")
        for prompt_type, prompts in eval_prompts.items():
            print(f"  {prompt_type.value}: {len(prompts)} prompts")
            
    except Exception as e:
        print(f" X Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()