import torch
from datasets import load_dataset
import copy

from utils import get_hidden_states, apply_weight_delta, load_model_and_tokenizer, load_data
from actsvd import ActSVD_orthogonal
from eval import evaluate_gsm8k, evaluate_humaneval
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# --- Configuration ---
MODEL_OPTIONS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
]
MODEL_NAME = MODEL_OPTIONS[0]

# --- BATCH SIZES ---
# Critical: Keep SVD_BATCH_SIZE = 1 to avoid truncating long reasoning chains.
SVD_BATCH_SIZE = 16       
# Optimization: Set EVAL_BATCH_SIZE higher (32/64) for fast GPU evaluation.
EVAL_BATCH_SIZE = 128      

# --- SVD SETTINGS ---
SVD_RANK_MATH = 2          # Rank 1: Targets the primary "Math Direction"
SVD_RANK_CODE = 2           # Rank 1: Targets the primary "Code Direction"
DATASET_FRACTION = 1     # Use 100% of data to ensure robust SVD calculation
EVAL_SAMPLES = 100           # Number of samples for scoring (None for full run)

# --- PRUNING STRENGTH ---
ALPHA = 0.1                 # <--- SCALING FACTOR (0.1 = Subtract 10% of the vector)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: Running on CPU. This will be extremely slow.")

    # 1. Load Model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # 2. Load Data
    # Load *training* data for SVD extraction
    gsm_train_data, humaneval_train_data = load_data(DATASET_FRACTION)

    # Load *test* data for Evaluation
    print("\n[1/2] Loading HumanEval (test split for eval)...")
    code_test_dataset = load_dataset("openai_humaneval", split="test")

    print("\n[2/2] Loading GSM8K (test split for eval)...")
    math_test_dataset = load_dataset("gsm8k", "main", split="test")

    # 3. Baseline Evaluation
    print(f"\n--- Running Baseline Evaluation (Batch Size: {EVAL_BATCH_SIZE}) ---")
    baseline_math_score = evaluate_gsm8k(
        model, tokenizer, math_test_dataset, 
        num_samples=EVAL_SAMPLES, 
        batch_size=EVAL_BATCH_SIZE
    )
    baseline_code_score = evaluate_humaneval(
        model, tokenizer, code_test_dataset, 
        num_samples=EVAL_SAMPLES, 
        batch_size=EVAL_BATCH_SIZE
    )
    # baseline_math_score=0
    # baseline_code_score=0

    # 4. Extract Hidden States
    print(f"\n--- Extracting Hidden States (Math) (Batch Size: {SVD_BATCH_SIZE}) ---")
    activations_math = get_hidden_states(
        model, 
        dataset=gsm_train_data, 
        tokenizer=tokenizer, 
        text_column='answer',       # Uses full reasoning chain
        batch_size=SVD_BATCH_SIZE
    )
    
    print(f"\n--- Extracting Hidden States (Code) (Batch Size: {SVD_BATCH_SIZE}) ---")
    activations_code = get_hidden_states(
        model, 
        dataset=humaneval_train_data, 
        tokenizer=tokenizer, 
        text_column='canonical_solution', # Uses full code solution
        batch_size=SVD_BATCH_SIZE
    )

    # 5. Compute Deltas
    print("\n--- Computing Orthogonal Deltas ---")
    math_deltas, code_deltas = ActSVD_orthogonal(
        model,
        activations_math,
        activations_code,
        rank_task1=SVD_RANK_MATH,
        rank_task2=SVD_RANK_CODE
    )

    # 6. Create Updated Models
    print(f"\n--- Applying Deltas to Create New Models (Alpha: {ALPHA}) ---")
    
    # Apply Math Ablation
    model_math_updated = copy.deepcopy(model)
    apply_weight_delta(model_math_updated, math_deltas, alpha=ALPHA) # <--- Uses Config
    print("Created Math-focused model (Math Ablated).")

     #7. Re-evaluation
    print("\n--- Running Evaluation on Updated Models ---")
    
    print("Evaluating Math-Updated Model:")
    updated_math_math_score = evaluate_gsm8k(
        model_math_updated, tokenizer, math_test_dataset, 
        num_samples=EVAL_SAMPLES, 
        batch_size=EVAL_BATCH_SIZE
    )

    updated_math_code_score = evaluate_humaneval(
        model_math_updated, tokenizer, code_test_dataset, 
        num_samples=EVAL_SAMPLES, 
        batch_size=EVAL_BATCH_SIZE
    )

    model_math_updated.delete()
    torch.cuda.empty_cache()

    # Apply Code Ablation
    model_code_updated = copy.deepcopy(model)
    apply_weight_delta(model_code_updated, code_deltas, alpha=ALPHA) # <--- Uses Config
    print("Created Code-focused model (Code Ablated).")
    
    
    print("Evaluating Code-Updated Model:")
    updated_code_math_score = evaluate_gsm8k(
        model_code_updated, tokenizer, math_test_dataset, 
        num_samples=EVAL_SAMPLES, 
        batch_size=EVAL_BATCH_SIZE
    )
    updated_code_code_score = evaluate_humaneval(
        model_code_updated, tokenizer, code_test_dataset, 
        num_samples=EVAL_SAMPLES, 
        batch_size=EVAL_BATCH_SIZE
    )

    # 8. Report Results
    print("\n--- Task Interference Analysis Results ---")
    print(f"Model: {MODEL_NAME}")
    print(f"SVD Rank 'r_math': {SVD_RANK_MATH}")
    print(f"SVD Rank 'r_code': {SVD_RANK_CODE}")
    print(f"Pruning Strength (Alpha): {ALPHA}")
    print("\n--- Scores (Higher is Better) ---")
    print("| Model               | GSM-8k (Math) Score | HumanEval (Code) Score |")
    print("|---------------------|---------------------|------------------------|")
    print(f"| Baseline            | {baseline_math_score:<19.4f} | {baseline_code_score:<22.4f} |")
    print(f"| Updated (Math)      | {updated_math_math_score:<19.4f} | {updated_math_code_score:<22.4f} |")
    print(f"| Updated (Code)      | {updated_code_math_score:<19.4f} | {updated_code_code_score:<22.4f} |")
    print("-----------------------------------------------------------------")