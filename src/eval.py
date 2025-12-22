import re
import torch
import multiprocessing
import sys
import io
import contextlib
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# --- GSM8K Helper Functions ---

def extract_answer_number(text: str) -> float:
    text = text.replace(',', '')
    pattern = r"(-?[\d]+(?:[\.][\d]+)?)"
    if "####" in text:
        text = text.split("####")[-1]
    matches = re.findall(pattern, text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def is_correct_math(pred_str: str, label_str: str) -> bool:
    pred_num = extract_answer_number(pred_str)
    label_num = extract_answer_number(label_str)
    if pred_num is None or label_num is None:
        return False
    return abs(pred_num - label_num) < 1e-6

# --- HumanEval Helper Functions ---

def clean_code_generation(generation: str, prompt: str) -> str:
    if generation.startswith(prompt):
        code = generation[len(prompt):]
    else:
        code = generation
    if "```python" in code:
        code = code.split("```python")[1]
        if "```" in code:
            code = code.split("```")[0]
    elif "```" in code:
        code = code.split("```")[0]
    return code

def _run_test_case(code_str: str, test_case: str, entry_point: str, queue):
    full_program = code_str + "\n\n" + test_case + f"\ncheck({entry_point})"
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            exec_globals = {}
            exec(full_program, exec_globals)
        queue.put("PASS")
    except Exception as e:
        queue.put(f"FAIL: {str(e)}")

def run_code_with_timeout(code_str: str, test_case: str, entry_point: str, timeout: int = 3):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_test_case, args=(code_str, test_case, entry_point, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    if not queue.empty():
        return queue.get() == "PASS"
    return False

# --- Batched Evaluation Functions ---

def evaluate_gsm8k(model, tokenizer, test_dataset, num_samples=None, batch_size=8):
    """
    Evaluates GSM8K with batching for GPU acceleration.
    """
    model.eval()
    device = model.device
    
    # Critical: Set padding side to left for Causal LM generation
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    if num_samples:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    def collate_fn(batch):
        return [b['question'] for b in batch], [b['answer'] for b in batch]
        
    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    correct_count = 0
    total = 0
    
    print(f"Evaluating GSM8K on {len(test_dataset)} samples (Batch size: {batch_size})...")
    
    for questions, references in tqdm(dataloader):
        batch_prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step. Put the final answer after '####'."},
                {"role": "user", "content": q}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        # Slice off input tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for pred, ref in zip(decoded_preds, references):
            if is_correct_math(pred, ref):
                correct_count += 1
            total += 1
            
    score = correct_count / total if total > 0 else 0
    print(f"GSM8K Accuracy: {score:.4f}")
    return score

def evaluate_humaneval(model, tokenizer, test_dataset, num_samples=None, batch_size=8):
    """
    Evaluates HumanEval with batching for GPU acceleration.
    """
    model.eval()
    device = model.device
    
    # Critical: Set padding side to left for Causal LM generation
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    if num_samples:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
        
    # Custom collator to bundle all necessary fields
    def collate_fn(batch):
        prompts = [b['prompt'] for b in batch]
        tests = [b['test'] for b in batch]
        entries = [b['entry_point'] for b in batch]
        return prompts, tests, entries

    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    passed_count = 0
    total = 0
    
    print(f"Evaluating HumanEval on {len(test_dataset)} samples (Batch size: {batch_size})...")
    
    for prompts, tests, entries in tqdm(dataloader):
        batch_inputs = []
        for p in prompts:
            messages = [
                 {"role": "system", "content": "You are a coding assistant. Complete the python function provided. Do not provide explanation, just valid python code."},
                 {"role": "user", "content": f"Complete this function:\n\n{p}"}
            ]
            batch_inputs.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Check correctness (sequential execution on CPU, but generation was parallel)
        for pred_code, prompt, test_case, entry_point in zip(decoded_preds, prompts, tests, entries):
            cleaned_body = clean_code_generation(pred_code, prompt)
            
            # Re-assemble valid code
            if "def " not in cleaned_body and "class " not in cleaned_body:
                 full_code = prompt + "\n" + cleaned_body
            else:
                 full_code = cleaned_body

            if run_code_with_timeout(full_code, test_case, entry_point):
                passed_count += 1
            total += 1

    score = passed_count / total if total > 0 else 0
    print(f"HumanEval Pass@1: {score:.4f}")
    return score