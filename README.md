# qwen2.5-7b-orpo-alignment

> ORPO (Odds Ratio Preference Optimization) alignment pipeline for Qwen2.5-7B using Unsloth QLoRA on Kaggle T4 GPU.

A high-performance **ORPO** alignment pipeline for
[Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), optimized for **Kaggle T4 x2 GPU** with [Unsloth](https://github.com/unslothai/unsloth).

ORPO combines supervised fine-tuning (SFT) and preference alignment into a **single training step**,
eliminating the need for a separate reference model or a two-stage SFT-then-DPO pipeline.

### Key Features

- **Single-stage alignment** -- ORPO merges SFT + preference optimization (no reference model needed)
- **Unsloth acceleration** -- 2x faster training, ~60% less VRAM via fused kernels
- **QLoRA** -- 4-bit NF4 quantization with LoRA r=16 on all linear layers
- **Kaggle-native** -- Runs on free Kaggle T4 x2 GPU within 12-hour session limits
- **Config-driven** -- All hyperparameters in a single YAML file
- **Full evaluation** -- IFEval, GSM8K, MATH, MMLU benchmarks via lm-evaluation-harness

---

## Architecture Overview

```
                    +------------------+
                    |   orpo_config    |
                    |     (.yaml)      |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v---------+        +---------v----------+
     |    SlimOrca       |        | orpo-dpo-mix-40k   |
     |  (518K SFT data)  |        | (40K pref. pairs)  |
     +--------+----------+        +---------+----------+
              |                             |
              |  data_prep.py               |
              |  - ChatML formatting        |
              |  - ShareGPT -> ChatML       |
              +-------------+---------------+
                            |
                   +--------v---------+
                   |  Merged Dataset   |
                   |  prompt/chosen/   |
                   |  rejected         |
                   +--------+----------+
                            |
                   +--------v---------+
                   |  train_orpo.py    |
                   |  - Unsloth 4-bit  |
                   |  - QLoRA r=16     |
                   |  - ORPOTrainer    |
                   +--------+----------+
                            |
              +-------------+-------------+
              |                           |
     +--------v---------+       +--------v---------+
     |  LoRA Adapters    |       |  Merged Model    |
     |  (lightweight)    |       |  (full weights)  |
     +-------------------+       +--------+---------+
                                          |
                                 +--------v---------+
                                 |  evaluate.sh      |
                                 |  IFEval, GSM8K,   |
                                 |  MATH, MMLU       |
                                 +-------------------+
```

---

## Hardware Requirements

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU | NVIDIA T4 (~14.5 GB usable, SM 7.5+) | T4 x2 on Kaggle |
| VRAM | 12 GB | 14.5 GB |
| System RAM | 16 GB | 32 GB |
| Disk | 30 GB free | 50 GB free |
| CUDA | 11.8+ | 12.8+ |
| Python | 3.10+ | 3.12 |

> **Note**: The P100 GPU (Pascal, SM 6.0) is **not compatible** with Unsloth.
> Use the T4 accelerator in Kaggle (Settings > Accelerator > GPU T4 x2).

---

## Project Structure

```
qwen2.5-7b-orpo-alignment/
|
|-- requirements.txt              # Pip dependencies
|
|-- configs/
|   +-- orpo_config.yaml          # All hyperparameters
|
|-- src/
|   |-- __init__.py
|   |-- utils.py                  # Config, seed, GPU diagnostics, Kaggle helpers
|   |-- data_prep.py              # Dataset loading, ChatML formatting, merging
|   +-- train_orpo.py             # Unsloth + QLoRA + ORPOTrainer
|
|-- scripts/
|   +-- evaluate.sh               # lm-evaluation-harness benchmarks
|
+-- README.md                     # This file
```

---

## Quickstart (Kaggle Notebook)

### 1. Setup

Upload the `src/` and `configs/` directories as a **Kaggle Dataset** (e.g., `orpo-pipeline-src`).
In your Kaggle notebook, select **GPU T4 x2** as the accelerator, then:

```python
# Cell 1: Install dependencies
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.28" "trl>=0.12.0" "peft" "accelerate" "bitsandbytes"
!pip install datasets pyyaml wandb
```

### 2. Initialize Pipeline

```python
# Cell 2: Setup
import sys
sys.path.insert(0, "/kaggle/input/orpo-pipeline-src/src")

from utils import initialize_pipeline

config = initialize_pipeline("/kaggle/input/orpo-pipeline-src/configs/orpo_config.yaml")
```

### 3. Prepare Data

```python
# Cell 3: Data preparation
from data_prep import prepare_orpo_dataset

train_ds, val_ds = prepare_orpo_dataset(config)
```

### 4. Train

```python
# Cell 4: ORPO training
from train_orpo import run_orpo_training

trainer = run_orpo_training(config, train_ds, val_ds)
```

### 5. Test Inference

```python
# Cell 5: Quick inference test
from train_orpo import test_inference

test_inference(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    prompt="Explain quantum entanglement in simple terms.",
)
```

### 6. Evaluate (Optional)

```python
# Cell 6: Run benchmarks
!pip install -q lm-eval langdetect immutabledict
!bash /kaggle/input/orpo-pipeline-src/scripts/evaluate.sh \
    /kaggle/working/orpo_merged_model \
    /kaggle/working/eval_results
```

---

## Configuration

All hyperparameters are centralized in `configs/orpo_config.yaml`. Key settings:

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `Qwen/Qwen2.5-7B` | Base model (not Instruct) |
| `model.max_seq_length` | `1024` | Maximum sequence length (T4 VRAM safe) |
| `model.dtype` | `float16` | T4 does not support bf16 |
| `model.load_in_4bit` | `true` | QLoRA 4-bit quantization |

### LoRA

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.r` | `16` | LoRA rank |
| `lora.lora_alpha` | `16` | Scaling factor (alpha/r = 1.0) |
| `lora.lora_dropout` | `0.0` | Recommended 0 for QLoRA |
| `lora.target_modules` | all 7 projections | q/k/v/o/gate/up/down_proj |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.learning_rate` | `5e-6` | ORPO learning rate |
| `training.beta` | `0.1` | Odds-ratio preference weight |
| `training.per_device_train_batch_size` | `1` | Per-GPU batch size (T4 VRAM safe) |
| `training.gradient_accumulation_steps` | `8` | Effective batch = 1 x 8 = 8 |
| `training.num_train_epochs` | `1` | Training epochs |
| `training.max_seq_length` | `1024` | Max sequence length for training |
| `training.optim` | `adamw_8bit` | 8-bit optimizer for VRAM |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.sft_subset_size` | `50000` | SlimOrca samples to use |
| `data.preference_subset_size` | `13000` | Preference pairs to sample |
| `data.val_split` | `0.05` | Validation split ratio |

---

## Data Pipeline

### Sources

| Dataset | Role | Format | Size |
|---------|------|--------|------|
| [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | SFT signal (chosen completions) | ShareGPT | ~518K |
| [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) | Preference pairs (chosen + rejected) | Chat messages | ~40K |

### ChatML Template (Qwen 2.5)

All data is formatted using the ChatML template that Qwen 2.5 expects:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

### Processing Steps

1. **SlimOrca**: Parse ShareGPT format (`from`/`value`) -> map roles (`human`->`user`, `gpt`->`assistant`) -> ChatML
2. **orpo-dpo-mix-40k**: Extract prompt (system+user messages) and response (assistant) from each conversation -> ChatML
3. **Merge**: Concatenate both processed datasets
4. **Filter**: Remove toxic samples (`toxic-dpo-v0.2`) and empty entries
5. **Split**: 95% train / 5% validation

---

## Evaluation Benchmarks

| Benchmark | Task | Few-shot | What it measures |
|-----------|------|----------|-----------------|
| IFEval | `ifeval` | 0-shot | Instruction following compliance |
| GSM8K | `gsm8k` | 5-shot | Multi-step math reasoning |
| MATH | `minerva_math` | 4-shot | Competition-level mathematics |
| MMLU | `mmlu` | 5-shot | Broad knowledge (57 subjects) |

Run with:
```bash
bash scripts/evaluate.sh /path/to/model /path/to/results
```

---

## VRAM Budget (T4, ~14.5 GB usable)

| Component | Estimated |
|-----------|-----------|
| Qwen2.5-7B in 4-bit (NF4) | ~4.5 GB |
| LoRA adapters (r=16) | ~0.2 GB |
| Unsloth optimized optimizer | ~1.0 GB |
| Activations (batch=1, seq=1024, grad ckpt) | ~1.5 GB |
| Gradient checkpointing savings | -0.5 GB |
| CUDA context + buffers | ~1.5 GB |
| **Total** | **~8.2 GB / 14.5 GB** |
| **Headroom** | **~6.3 GB** |

---

## Kaggle Secrets

Add these as Kaggle Secrets (Settings > Add Secret) for full functionality:

| Secret Name | Required | Purpose |
|-------------|----------|---------|
| `HF_TOKEN` | Optional | Access gated HuggingFace models |
| `WANDB_API_KEY` | Optional | Enable WandB cloud logging |

If WandB is not configured, logging falls back to offline mode automatically.

---

## Output Files

After training completes, the following are saved to `/kaggle/working/`:

```
/kaggle/working/
|-- processed_data/           # Preprocessed Arrow datasets
|   |-- train/
|   +-- val/
|-- orpo_checkpoints/         # Trainer checkpoints (last 2)
|-- orpo_lora_adapters/       # Lightweight LoRA adapter files
|-- orpo_merged_model/        # Full merged model (base + LoRA)
+-- eval_results/             # Benchmark JSON results + logs
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unsloth** over vanilla HF | 2x training speed, ~60% less VRAM via fused kernels |
| **Base model** (not Instruct) | ORPO handles SFT + alignment jointly; avoids re-aligning |
| **fp16** (not bf16) | T4 GPU (SM 7.5) does not support bf16 |
| **r=16** LoRA rank | Good capacity/VRAM balance for 7B params |
| **adamw_8bit** optimizer | Saves ~40% optimizer state VRAM |
| **Gradient checkpointing** | Trades compute for ~30% activation VRAM savings |
| **Toxic data filtering** | Removes `toxic-dpo-v0.2` from preference set |
| **YAML config** | Decouples hyperparameters from code for easy sweeps |

---

## Troubleshooting

### Common Issues

**"dtype mismatch" or "bf16 not supported"**
- Ensure `training.bf16: false` and `training.fp16: true` in the config.
- T4 GPUs do not support bfloat16.

**Out of Memory (OOM)**
- Reduce `training.per_device_train_batch_size` to 1.
- Reduce `model.max_seq_length` to 1024.
- Ensure `lora.use_gradient_checkpointing: "unsloth"` is set.

**Unsloth crashes on GPU**
- Verify you are using a T4 GPU, not P100.
- Check: `nvidia-smi` should show "Tesla T4".

**HuggingFace gated model access denied**
- Add your `HF_TOKEN` as a Kaggle Secret.
- Accept the model license on HuggingFace.

**WandB not logging**
- Add `WANDB_API_KEY` as a Kaggle Secret.
- Or set `wandb.enabled: false` in config for offline mode.

---

## References

- [ORPO Paper](https://arxiv.org/abs/2403.07691) - Hong et al., 2024
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [TRL ORPOTrainer](https://huggingface.co/docs/trl/main/en/orpo_trainer) - HuggingFace TRL
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B) - Alibaba Cloud
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - EleutherAI
- [Fine-tune Llama 3 with ORPO](https://huggingface.co/blog/mlabonne/orpo-llama-3) - mlabonne

---

## License

This pipeline code is provided as-is for research and educational purposes.
Model weights are subject to their respective licenses (Qwen2.5: Apache 2.0).
