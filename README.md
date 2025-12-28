# Mechanistic-Interpretability-for-Coding-vs-Math-Reasoning-Traces
Full Technical Report : https://docs.google.com/document/d/1Ek-Gk5BzIZbHhdbvILMrvcLIdM8hPW6WTlft7k49pIM/
Minimal instructions to install dependencies and run the analysis driver `src/main.py`.

**Requirements**
- Python 3.13.x (recommended).
- A GPU with sufficient memory (the default model is `Qwen/Qwen2.5-1.5B-Instruct`).
- Hugging Face access token if the model requires authentication.

**Install (recommended: conda)**

1. Create and activate an environment (conda example):

```bash
conda create -n mechint python=3.13 -y
conda activate mechint
pip install -r requirements.txt
```

2. Alternatively, use a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Hugging Face authentication (if required)**

If the model is gated or requires login, run:

```bash
pip install huggingface-hub
huggingface-cli login
```

Or export a token:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_..."
```

**Run Example:**

```bash
python src/main.py
```

Notes:
- The script uses a small in-file configuration at the top of `src/main.py` (model choice, batch sizes, SVD ranks, alpha, etc.). Edit those variables if you want to change behavior.
- If you run on CPU or a single GPU with limited memory, the script will warn and run much slower.

**Quick configuration options (in `src/main.py`)**
- `MODEL_NAME`: replace with a different HF model identifier.
- `SVD_BATCH_SIZE`: keep as `1` for very long sequences; set higher (e.g., 16) to speed up extraction when memory allows.
- `EVAL_BATCH_SIZE`: larger value speeds up evaluation if GPU memory permits.
- `SVD_RANK_MATH`, `SVD_RANK_CODE`: SVD ranks used for ActSVD extraction.
- `DATASET_FRACTION`: use a fraction <1.0 for quick testing.
- `EVAL_SAMPLES`: number of samples to use for evaluation (set small to test quickly).
- `ALPHA`: scaling factor applied when modifying model weights.
