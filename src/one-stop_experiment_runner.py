# src/one_stop_experiment_runner.py


import json
import os
import sys
import importlib.util
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_module_from_file(filepath, module_name):
    """Dynamically load a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OneStopExperimentRunner:
    """
    One-stop runner that executes all three evaluation stages:
    1. Experiment execution
    2. Code-based evaluation  
    3. LLM judge evaluation
    """
    
    def __init__(self, base_vector_index_dir: str = "vector_index"):
        self.base_vector_index_dir = base_vector_index_dir
        self.results_dir = "evaluation/results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_complete_pipeline(self, experiment_sets: List[str] = None) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            experiment_sets: List of experiment sets to run
            
        Returns:
            Complete pipeline results
        """
        print("=" * 80)
        print("One-Stop Experiment Runner - Complete Evaluation Pipeline")
        print("=" * 80)
        
        # Stage 1: Run experiments
        print("\n" + "="*50)
        print("STAGE 1: Running RAG Experiments")
        print("="*50)
        
        experiment_results = self._run_experiments(experiment_sets)
        if not experiment_results:
            print("ERROR: No experiment results generated. Exiting pipeline.")
            return {}
        
        # Stage 2: Code-based evaluation
        print("\n" + "="*50)
        print("STAGE 2: Code-Based Evaluation")
        print("="*50)
        
        code_eval_results = self._run_code_evaluation()
        if not code_eval_results:
            print("ERROR: Code evaluation failed. Exiting pipeline.")
            return {}
        
        # Stage 3: LLM Judge evaluation
        print("\n" + "="*50)
        print("STAGE 3: LLM Judge Evaluation")
        print("="*50)
        
        llm_eval_results = self._run_llm_judge_evaluation()
        
        # Combine results
        final_results = {
            "pipeline_stages": {
                "experiments": "completed",
                "code_evaluation": "completed", 
                "llm_judge_evaluation": "completed" if llm_eval_results else "skipped"
            },
            "output_files": self._get_output_files()
        }
        
        # Save final summary
        self._save_final_summary(final_results)
        
        print("\n" + "="*80)
        print("COMPLETE: All evaluation stages finished successfully!")
        print("="*80)
        
        return final_results
    
    def _run_experiments(self, experiment_sets: List[str] = None) -> bool:
        """Run experiment stage using existing experiment_runner.py"""
        try:
            # Import and run the experiment runner
            experiment_runner_path = os.path.join(project_root, "src", "experiment_runner.py")
            exp_runner_module = load_module_from_file(experiment_runner_path, "experiment_runner")
            
            # Run the main function from experiment_runner
            exp_runner_module.main()
            
            # Check if results were generated
            results_file = os.path.join(self.results_dir, "all_experiments_results.json")
            if os.path.exists(results_file):
                print(f" V Experiment results saved: {results_file}")
                return True
            else:
                print(" X No experiment results file found")
                return False
                
        except Exception as e:
            print(f" X Error running experiments: {e}")
            return False
    
    def _run_code_evaluation(self) -> bool:
        """Run code-based evaluation using existing evaluator.py"""
        try:
            # Import and run the evaluator
            evaluator_path = os.path.join(project_root, "evaluation", "evaluator.py")
            evaluator_module = load_module_from_file(evaluator_path, "evaluator")
            
            # Run the main function from evaluator
            evaluator_module.main()
            
            # Check if results were generated
            code_eval_dir = os.path.join(self.results_dir, "code_evaluator_results")
            results_file = os.path.join(code_eval_dir, "evaluator_all_experiments.json")
            if os.path.exists(results_file):
                print(f" V Code evaluation results saved: {results_file}")
                return True
            else:
                print(" X No code evaluation results file found")
                return False
                
        except Exception as e:
            print(f" X Error running code evaluation: {e}")
            return False
    
    def _run_llm_judge_evaluation(self) -> bool:
        """Run LLM judge evaluation using existing llm_as_judge.py"""
        try:
            # Import and run the LLM judge
            llm_judge_path = os.path.join(project_root, "evaluation", "llm_as_judge.py")
            llm_judge_module = load_module_from_file(llm_judge_path, "llm_as_judge")
            
            # Run the main function from llm_as_judge
            llm_judge_module.main()
            
            # Check if results were generated
            llm_eval_dir = os.path.join(self.results_dir, "llm_as_judge_results")
            results_file = os.path.join(llm_eval_dir, "llm_judge_all_experiments.json")
            if os.path.exists(results_file):
                print(f" V LLM judge results saved: {results_file}")
                return True
            else:
                print(" X No LLM judge results file found")
                return False
                
        except Exception as e:
            print(f" X Error running LLM judge evaluation: {e}")
            print("Note: This may be due to missing transformers library or model")
            return False
    
    def _get_output_files(self) -> Dict[str, List[str]]:
        """Get list of all output files generated by the pipeline"""
        output_files = {}
        
        # Experiment results
        exp_files = []
        for file in os.listdir(self.results_dir):
            if file.endswith('.json') and not file.startswith('complete_'):
                exp_files.append(file)
        output_files["experiments"] = exp_files
        
        # Code evaluation results
        code_eval_dir = os.path.join(self.results_dir, "code_evaluator_results")
        if os.path.exists(code_eval_dir):
            code_files = [f for f in os.listdir(code_eval_dir) if f.endswith('.json')]
            output_files["code_evaluation"] = code_files
        
        # LLM judge results
        llm_eval_dir = os.path.join(self.results_dir, "llm_as_judge_results")
        if os.path.exists(llm_eval_dir):
            llm_files = [f for f in os.listdir(llm_eval_dir) if f.endswith('.json')]
            output_files["llm_judge_evaluation"] = llm_files
        
        return output_files
    
    def _save_final_summary(self, results: Dict[str, Any]):
        """Save final pipeline summary"""
        summary_file = os.path.join(self.results_dir, "complete_pipeline_summary.json")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"V Pipeline summary saved: {summary_file}")
    
    def print_pipeline_summary(self):
        """Print summary of the pipeline execution and outputs"""
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        output_files = self._get_output_files()
        
        print("\nGenerated Files:")
        for stage, files in output_files.items():
            print(f"\n{stage.replace('_', ' ').title()}:")
            for file in files:
                print(f"  - {file}")
        
        # Count total experiments and evaluations
        total_exp_files = len(output_files.get("experiments", []))
        total_code_files = len(output_files.get("code_evaluation", []))
        total_llm_files = len(output_files.get("llm_judge_evaluation", []))
        
        print(f"\nTotal Files Generated:")
        print(f"  - Experiments: {total_exp_files}")
        print(f"  - Code Evaluations: {total_code_files}")
        print(f"  - LLM Judge Evaluations: {total_llm_files}")
        
        # Check for main result files
        main_files = {
            "Raw Experiments": "all_experiments_results.json",
            "Code Evaluation": "code_evaluator_results/evaluator_all_experiments.json", 
            "LLM Judge": "llm_as_judge_results/llm_judge_all_experiments.json"
        }
        
        print(f"\nMain Result Files Status:")
        for name, path in main_files.items():
            full_path = os.path.join(self.results_dir, path)
            status = " V FOUND" if os.path.exists(full_path) else "X MISSING"
            print(f"  - {name}: {status}")


def main():
    """Main function to run the complete one-stop experiment pipeline"""
    print("Hollow Knight RAG One-Stop Experiment Runner")
    print("This will run: Experiments → Code Evaluation → LLM Judge Evaluation")
    print("=" * 70)
    
    # Initialize runner
    runner = OneStopExperimentRunner()
    
    # Run complete pipeline
    results = runner.run_complete_pipeline()
    
    if results:
        # Print summary
        runner.print_pipeline_summary()
        
        print("\nHAPPY ENDING: PIPELINE COMPLETED SUCCESSFULLY!")
        print("All evaluation stages have been executed.")
        print(f"Results are available in: {runner.results_dir}/")
    else:
        print("\n X PIPELINE FAILED!")
        print("Some stages may not have completed successfully.")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    main()