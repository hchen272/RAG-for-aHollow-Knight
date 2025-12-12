# src/interactive_rag_system.py
"""
Interactive RAG System for Elden Ring Wiki Q&A
Users can interact with the RAG system by entering questions through the console
"""

import os
import sys
from typing import List, Dict, Any
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from hollow_knight_rag_system import HollowKnightRAGSystem

class InteractiveRAGSystem(HollowKnightRAGSystem):
    """
    Interactive RAG System:
    Allow user to input questions
    """
    
    def __init__(self, 
                 base_vector_index_dir: str = "vector_index",
                 retriever_name: str = "bge_small",
                 model_name: str = "google/flan-t5-large"):
        """
        Initialize RAG System
        
        Args:
            base_vector_index_dir: Vector Index Directory
            retriever_name: Searcher Name
            model_name: Name of the generative model
        """
        # Initialize the parent class using the vanilla_rag prompt template
        super().__init__(
            base_vector_index_dir=base_vector_index_dir,
            retriever_name=retriever_name,
            model_name=model_name,
            prompt_type="instruction_tuned_rag"
        )
        print(" V Interactive RAG system initialized")
    
    def enhanced_query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Enhanced query method that returns detailed response information

        Args:
        question: User question
        k: Number of documents used to generate the answer

        Returns:
        A response dictionary containing detailed information
        """
        # Step 1: Retrieve the relevant documents
        all_found_docs, retrieved_docs = self.retrieve(question, k=k)
        
        # Step 2: Prepare the context for generating the answer
        if retrieved_docs:
            context_for_generation = self.format_context(retrieved_docs)
            docs_for_generation = retrieved_docs
        else:
            context_for_generation = ""
            docs_for_generation = []
        
        # Step 3: Build the prompt and generate the answer
        prompt = self.build_prompt(question, context_for_generation)
        answer = self.generate_answer(prompt)
        
        # Step 4: Format all documents for display
        all_docs_formatted = self.format_context(all_found_docs)
        
        # Calculate confidence rate
        confidence = 0.0
        if all_found_docs:
            confidence = min(np.mean([doc["score"] for doc in all_found_docs]), 1.0)
        
        # structure response
        response = {
            "question": question,
            "answer": answer,
            "all_docs_formatted": all_docs_formatted,
            "confidence": float(confidence),
            "sources_count": len(docs_for_generation),
            "retriever_used": self.retriever_name,
            "documents_retrieved": len(all_found_docs),
            "documents_used": len(docs_for_generation)
        }
        
        return response

def display_response(response: Dict[str, Any]):
    """
    Display in user-friendly form
    
    Args:
    response: Response dictionary of the RAG system
    """
    print("\n" + "="*70)
    print("RESPONSE")
    print("="*70)
    print(f" ? Question: {response['question']}")
    print(f"-> Answer: {response['answer']}")
    print(f"-> Confidence: {response['confidence']:.3f}")
    print(f"-> Sources used: {response['sources_count']}")
    print(f"-> Documents retrieved: {response['documents_retrieved']}")
    print(f"-> Retriever: {response['retriever_used']}")
    
    # If there is availible documents, show first three
    if response['all_docs_formatted'] and response['all_docs_formatted'] != "No specific information found in the Howllow Knight Dataset.":
        print("\n  -> Top Retrieved Documents:")
        docs = response['all_docs_formatted'].split('\n\n')[:3]  # show first three
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc[:150]}..." if len(doc) > 150 else f"   {i}. {doc}")
    print("="*70 + "\n")

def main():
    """
    Main interactive function
    """
    print("   Howllow Knight - Interactive RAG System")
    print("="*50)
    print("Configuration:")
    print(" o Retriever: bge_small")
    print(" o Prompt: vanilla_rag") 
    print(" o Generator: google/flan-t5-large")
    print("="*50)
    
    try:
        # Initialize RAG system
        print(" o Initializing RAG system...")
        rag_system = InteractiveRAGSystem(
            base_vector_index_dir="vector_index",
            retriever_name="miniLM",
            model_name="google/flan-t5-large"
        )
        
        print(" V System ready! You can now ask questions about Elden Ring.")
        print(" HINT: Type 'quit', 'exit', or 'q' to end the session.")
        print("-" * 50)
        
        # Interactive loop
        while True:
            try:
                # get input
                question = input("\no Your question: ").strip()
                
                # check quit
                if question.lower() in ['quit', 'exit', 'q']:
                    print(" Thank you for using the Elden Ring Wiki RAG System!")
                    break
                
                if not question:
                    print("! Please enter a question ! ")
                    continue
                
                # error dealing
                print(" o Searching for information...")
                response = rag_system.enhanced_query(question, k=5)
                
                # display response
                display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n Session interrupted. Thank you for using the system!")
                break
            except Exception as e:
                print(f" X Error processing your question: {e}")
                print("Please try again with a different question.")
                
    except Exception as e:
        print(f" X Failed to initialize RAG system: {e}")
        print("Please check if the vector index and models are properly set up.")

if __name__ == "__main__":
    main()