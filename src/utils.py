import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from tqdm.auto import tqdm

def load_model_and_tokenizer(model_name, device='cuda'):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def load_data(dataset_fraction):    
    # GSM-8K (Math)
    gsm_data_raw = load_dataset("gsm8k", "main", split="train")
    gsm_data_raw = gsm_data_raw.filter(lambda x: x['answer'] and len(x['answer'].strip()) > 0)
    gsm_subset = gsm_data_raw.select(range(int(dataset_fraction*len(gsm_data_raw))))

    # OpenAI HumanEval (Code)
    humaneval_data_raw = load_dataset("openai_humaneval", split="test")
    humaneval_subset = humaneval_data_raw.select(range(int(dataset_fraction*len(humaneval_data_raw))))
    
    print(f"Using {len(gsm_subset)} samples for gsm8k activations (from train split).")
    print(f"Using {len(humaneval_subset)} samples for humaneval activations (from train split).")
    
    return gsm_subset, humaneval_subset

def get_hidden_states(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    text_column: str,
    batch_size: int = 1
) -> List[List[torch.Tensor]]:
    """
    Collects hidden state activations.
    """
    device = model.device
    model.eval()
        
    print(f"Extracting hidden states from '{text_column}'...")
    
    def collate_fn(batch):
        try:
            texts = [ex[text_column] for ex in batch]
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return None 
            
            tokenized_batch = tokenizer(valid_texts, padding=False, truncation=False)
        except Exception:
            return None
            
        # Find min length to avoid padding issues (simple truncation strategy)
        try:
            seq_lengths = [len(ids) for ids in tokenized_batch['input_ids']]
            valid_lengths = [l for l in seq_lengths if l > 0]
            if not valid_lengths:
                return None
            min_len = min(valid_lengths)
        except Exception:
             return None
        
        all_input_ids = []
        all_attn_masks = []
        
        for i in range(len(tokenized_batch['input_ids'])):
            if len(tokenized_batch['input_ids'][i]) >= min_len:
                input_ids = tokenized_batch['input_ids'][i][:min_len]
                attention_mask = tokenized_batch['attention_mask'][i][:min_len]
                all_input_ids.append(torch.tensor(input_ids))
                all_attn_masks.append(torch.tensor(attention_mask))
        
        if not all_input_ids:
            return None

        return {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attn_masks)
        }
    
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    all_hidden_states_by_layer = None

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting hidden states"):
            if batch is None: 
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True
            )
            
            batch_hidden_states = outputs.hidden_states 
            
            if all_hidden_states_by_layer is None:
                num_layers = len(batch_hidden_states)
                all_hidden_states_by_layer = [[] for _ in range(num_layers)]
            
            for i in range(len(batch_hidden_states)):
                layer_state = batch_hidden_states[i]
                sequences = layer_state.unbind(dim=0)
                for seq in sequences:
                    all_hidden_states_by_layer[i].append(seq.cpu())

    if all_hidden_states_by_layer is None:
        print("Warning: Dataset was empty or no data was processed.")
        return []

    return all_hidden_states_by_layer

def get_target_modules(model: AutoModelForCausalLM) -> List[Dict]:
    target_modules = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        base_layer_prefix = "model.layers"
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        base_layer_prefix = "transformer.h"
    else:
        print("Error: Unknown model structure.")
        return []

    print(f"Found {len(layers)} layers at `{base_layer_prefix}`. Detecting MLP modules...")
        
    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, 'mlp'):
            mlp_block = layer.mlp
            mlp_prefix = f"{base_layer_prefix}.{layer_idx}.mlp"
        elif hasattr(layer, 'ffn'):
            mlp_block = layer.ffn
            mlp_prefix = f"{base_layer_prefix}.{layer_idx}.ffn"
        else:
            continue
            
        for name, module in mlp_block.named_modules():
            if isinstance(module, nn.Linear):
                # Filter out output projections (down_proj) because inputs don't match hidden size
                if "down" in name or "fc2" in name or "c_proj" in name:
                    continue
                    
                full_name = f"{mlp_prefix}.{name}"
                if '.' not in name:
                    target_modules.append({
                        "name": full_name,
                        "module": module,
                        "input_layer_idx": layer_idx
                    })
                    
    return target_modules

def apply_weight_delta(model: nn.Module, delta_weights: Dict[str, torch.Tensor], alpha: float = 1.0):
    """
    Applies an additive "delta" to the model's existing weights with a scaling factor.
    """
    print(f"Applying weight deltas with scale {alpha}...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in delta_weights:
                try:
                    delta = delta_weights[name].to(param.device, dtype=param.dtype)
                    # Subtract/Add the scaled delta
                    param.data += (delta * alpha)
                except Exception as e:
                    print(f"Error applying delta for '{name}': {e}")
    print("Finished applying weight deltas.")