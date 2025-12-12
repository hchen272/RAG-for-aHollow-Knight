# src/multi_index_builder.py

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add configs to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_index import HollowKnightVectorIndexBuilder
from configs.retriever_configs import get_all_retriever_names, get_retriever_config

logger = logging.getLogger(__name__)

class MultiIndexBuilder:
    """
    Builds vector indices for all configured retriever models
    Handles parallel index building and verification
    """
    
    def __init__(self, base_index_dir: str = "../vector_index"):
        self.base_index_dir = Path(base_index_dir)
        self.logger = logging.getLogger(__name__)
        
    def build_all_indices(self, data_path: str, specific_retrievers: List[str] = None) -> Dict[str, Any]:
        """
        Build indices for all configured retriever models
        
        Args:
            data_path: Path to formatted data JSON
            specific_retrievers: List of specific retrievers to build (None for all)
            
        Returns:
            Dictionary with build results and statistics
        """
        # Get retrievers to build
        if specific_retrievers is None:
            retrievers = get_all_retriever_names()
        else:
            retrievers = specific_retrievers
        
        self.logger.info(f"Building indices for {len(retrievers)} retrievers")
        
        results = {}
        successful_builds = 0
        
        for retriever_name in retrievers:
            try:
                self.logger.info(f"Building index for {retriever_name}")
                
                # Get retriever configuration
                config = get_retriever_config(retriever_name)
                
                # Build index path
                index_path = self.base_index_dir / f"hollow_knight_{retriever_name}"
                
                # Build index with new JSON format
                builder = HollowKnightVectorIndexBuilder(config["model_name"])
                vector_index = builder.build_index(
                    data_path=data_path,
                    output_path=str(index_path),
                    dimension=config["dimension"],
                )
                
                # Store result
                results[retriever_name] = {
                    "status": "success",
                    "index_path": str(index_path),
                    "model": config["model_name"],
                    "dimension": config["dimension"],
                    "description": config["description"]
                }
                successful_builds += 1
                
                self.logger.info(f"Successfully built index for {retriever_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to build index for {retriever_name}: {e}")
                results[retriever_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate summary
        summary = {
            "total_retrievers": len(retrievers),
            "successful_builds": successful_builds,
            "failed_builds": len(retrievers) - successful_builds,
            "results": results
        }
        
        return summary
    
    def verify_all_indices(self) -> Dict[str, Any]:
        """
        Verify all built indices
        
        Returns:
            Dictionary with verification results
        """
        retrievers = get_all_retriever_names()
        verification_results = {}
        
        for retriever_name in retrievers:
            index_path = self.base_index_dir / f"hollow_knight_{retriever_name}"
            
            try:
                from vector_index import VectorIndex
                
                if not os.path.exists(index_path / "index.faiss"):
                    verification_results[retriever_name] = {
                        "status": "not_found",
                        "message": "Index files not found"
                    }
                    continue
                
                # Load and verify index
                vector_index = VectorIndex(str(index_path))
                vector_index.load()
                
                stats = vector_index.get_stats()
                
                verification_results[retriever_name] = {
                    "status": "verified",
                    "total_vectors": stats["total_vectors"],
                    "dimension": stats["dimension"],
                    "document_types": stats["document_types"]
                }
                
                self.logger.info(f"Verified index for {retriever_name}: {stats['total_vectors']} vectors")
                
            except Exception as e:
                verification_results[retriever_name] = {
                    "status": "verification_failed",
                    "error": str(e)
                }
                self.logger.error(f"Failed to verify index for {retriever_name}: {e}")
        
        return verification_results
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about all built indices
        
        Returns:
            Dictionary with index information
        """
        retrievers = get_all_retriever_names()
        index_info = {}
        
        for retriever_name in retrievers:
            index_path = self.base_index_dir / f"hollow_knight_{retriever_name}"
            config_path = index_path / "config.json"
            
            info = {
                "path": str(index_path),
                "exists": os.path.exists(index_path)
            }
            
            if info["exists"] and os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    info.update(config)
                except:
                    pass
            
            index_info[retriever_name] = info
        
        return index_info


def build_all_hollow_knight_indices(
    data_path: str = "../data/hollow_knight_fandom_rag_optimized.json",
    base_index_dir: str = "../vector_index",
    specific_retrievers: List[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to build indices for all configured retrievers
    
    Args:
        data_path: Path to formatted data JSON
        base_index_dir: Base directory for index storage
        specific_retrievers: List of specific retrievers to build
        
    Returns:
        Build results summary
    """
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print("Building Hollow Knight Vector Indices for All Retrievers")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Output Base: {base_index_dir}")
    
    # Get retrievers to build
    if specific_retrievers is None:
        retrievers = get_all_retriever_names()
    else:
        retrievers = specific_retrievers
    
    print(f"Retrievers: {', '.join(retrievers)}")
    print()
    
    # Build indices
    multi_builder = MultiIndexBuilder(base_index_dir)
    results = multi_builder.build_all_indices(data_path, retrievers)
    
    # Print summary
    print("\nBuild Summary")
    print("=" * 30)
    print(f"Total: {results['total_retrievers']}")
    print(f"Successful: {results['successful_builds']}")
    print(f"Failed: {results['failed_builds']}")
    
    # Print details
    print("\nDetails")
    print("=" * 30)
    for retriever, result in results["results"].items():
        status_icon = "V" if result["status"] == "success" else "X"
        print(f"{status_icon} {retriever}: {result['status']}")
        if result["status"] == "success":
            print(f"   Model: {result['model']}")
            print(f"   Path: {result['index_path']}")
        elif "error" in result:
            print(f"   Error: {result['error']}")
    
    return results


def main():
    """Main function for building all indices"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Hollow Knight vector indices for all retrievers")
    parser.add_argument("--data", default="data/hollow_knight_fandom_rag_optimized.json",
                       help="Path to formatted data JSON")
    parser.add_argument("--output", default="vector_index",
                       help="Base output directory for indices")
    parser.add_argument("--retrievers", nargs="+",
                       help="Specific retrievers to build (default: all)")
    
    args = parser.parse_args()
    
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Build all indices
        results = build_all_hollow_knight_indices(
            data_path=args.data,
            base_index_dir=args.output,
            specific_retrievers=args.retrievers
        )
        
        # Verify indices if any were built successfully
        successful_builds = results["successful_builds"]
        if successful_builds > 0:
            print(f"\nVerifying {successful_builds} indices...")
            multi_builder = MultiIndexBuilder(args.output)
            verification_results = multi_builder.verify_all_indices()
            
            print("\nVerification Results")
            print("=" * 30)
            for retriever, result in verification_results.items():
                if result["status"] == "verified":
                    print(f"V {retriever}: {result['total_vectors']} vectors")
                else:
                    print(f"X {retriever}: {result['status']}")
        
        print(f"\nAll operations completed!")
        
    except Exception as e:
        print(f"Failed to build indices: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()