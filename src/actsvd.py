import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from utils import get_target_modules

def get_projection_for_seq(
    seq_activations: torch.Tensor, 
    rank: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Computes P_i = V_r_i @ V_r_i.T for a single activation tensor.
    Returns a (D_in, D_in) projection matrix, or None if SVD fails.
    """
    if seq_activations.shape[0] < rank: # Skip if not enough tokens
        return None
        
    A = seq_activations.to(device, dtype=torch.float32) # (N_tokens, D_hidden)
    
    try:
        # SVD(A) -> A = U S Vh
        # Vh is (K, D_hidden), where K = min(N_tokens, D_hidden)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        
        # V_r = Vh.T[:, :rank]
        # We take the first 'r' rows of Vh and transpose them
        V_r = Vh[:rank, :].T # Shape: (D_hidden, r)
        
        # P_i = V_r @ V_r.T
        P_i = V_r @ V_r.T # Shape: (D_hidden, D_hidden)
        
        del A, U, S, Vh, V_r
        return P_i
    
    except Exception as e:
        # print(f"Warning: SVD failed for a sequence: {e}")
        del A # Free memory
        return None
    
def ActSVD_orthogonal(
    model: AutoModelForCausalLM,
    activations_task1: List[List[torch.Tensor]], # List[layer][sequence]
    activations_task2: List[List[torch.Tensor]], # List[layer][sequence]
    rank_task1: int,
    rank_task2: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Calculates the orthogonal task-specific deltas for MLP layers
    based on the *average delta* (Delta_avg = Avg(Delta_i)).

    Based on the formulas:
    Delta_1_i = - W @ (I - P_2_i) @ P_1_i
    Delta_2_i = - W @ (I - P_1_i) @ P_2_i
    
    Then the final delta for the module is Avg(Delta_1_i) over all sequences.
    """
    
    DEVICE = model.device
    DTYPE = model.dtype # Use model's dtype
    
    target_modules = get_target_modules(model)
    
    delta_weights_task1_focused = {}
    delta_weights_task2_focused = {}

    print("--- Calculating Orthogonal Deltas (Average per-sequence delta) ---")
    
    # Loop over each target module (e.g., gate_proj, fc1)
    for mod_info in tqdm(target_modules, desc="Averaging deltas per module"):
        layer_idx = mod_info["input_layer_idx"]
        param_name = mod_info["name"] + ".weight"
        
        # Get the original weight, cast to float32 for precision
        W = mod_info["module"].weight.data.to(DEVICE, dtype=torch.float32) # (D_out, D_in)
        D_in = W.shape[1]
        I = torch.eye(D_in, device=DEVICE, dtype=torch.float32)
        
        # Initialize running sums for the average
        delta_sum_task1 = torch.zeros_like(W, device=DEVICE, dtype=torch.float32)
        delta_sum_task2 = torch.zeros_like(W, device=DEVICE, dtype=torch.float32)
        num_valid_seqs = 0
        
        # Get the lists of activations for *this layer*
        seq_activations_task1 = activations_task1[layer_idx]
        seq_activations_task2 = activations_task2[layer_idx]
        
        num_seqs = min(len(seq_activations_task1), len(seq_activations_task2))

        # Loop over *every sequence* for this module
        for i in range(num_seqs):
            
            # 1. Get activations for sequence i
            A_1_i = seq_activations_task1[i]
            A_2_i = seq_activations_task2[i]

            # 2. Compute P_1_i
            P_1_i = get_projection_for_seq(A_1_i, rank_task1, DEVICE)
            if P_1_i is None:
                continue # Skip sequence if SVD failed or too short
            
            # 3. Compute P_2_i
            P_2_i = get_projection_for_seq(A_2_i, rank_task2, DEVICE)
            if P_2_i is None:
                continue # Skip sequence
                
            # We now have W, I, P_1_i, and P_2_i
            
            # --- 4. Calculate Task 1 (e.g., Math) Delta for seq i ---
            # Delta_1_i = W @ (I - P_2_i) @ P_1_i
            # Note: We store the *negative* delta (for addition)
            Delta_1_i = W @ (I - P_2_i) @ P_1_i
            delta_sum_task1 += -Delta_1_i
        
            # --- 5. Calculate Task 2 (e.g., Code) Delta for seq i ---
            # Delta_2_i = W @ (I - P_1_i) @ P_2_i
            Delta_2_i = W @ (I - P_1_i) @ P_2_i
            delta_sum_task2 += -Delta_2_i
            
            num_valid_seqs += 1
            
            # Clear per-sequence tensors
            del P_1_i, P_2_i, Delta_1_i, Delta_2_i
        
        # --- 6. Finalize Average Deltas for this module ---
        if num_valid_seqs > 0:
            delta_weights_task1_focused[param_name] = (delta_sum_task1 / num_valid_seqs).to(dtype=DTYPE)
            delta_weights_task2_focused[param_name] = (delta_sum_task2 / num_valid_seqs).to(dtype=DTYPE)
        else:
            print(f"Warning: No valid sequences found for module {param_name}. Skipping update.")

        # Clear cache
        del W, I, delta_sum_task1, delta_sum_task2
        torch.cuda.empty_cache()

    print("Delta calculations complete.")
    return delta_weights_task1_focused, delta_weights_task2_focused