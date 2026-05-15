"""
train_orpo.py -- ORPO training architecture for Qwen2.5-7B.

Implements:
  1. Unsloth FastLanguageModel for 4-bit model loading (T4-optimized)
  2. QLoRA adapter configuration (r=16, all linear projections)
  3. ORPOTrainer with ORPOConfig from TRL
  4. Gradient checkpointing (Unsloth mode)
  5. WandB integration (optional)
  6. LoRA adapter saving + optional merge with base model

Usage:
    from train_orpo import run_orpo_training
    trainer = run_orpo_training(config, train_dataset, val_dataset)

Target Hardware: Kaggle T4 x2 (Turing SM 7.5, 16GB VRAM, fp16 only)
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading (Unsloth 4-bit)
# =============================================================================

def load_model_and_tokenizer(
    config: Dict[str, Any],
) -> Tuple[Any, Any]:
    """
    Load the base model using Unsloth's FastLanguageModel with 4-bit quantization.

    This leverages Unsloth's optimized loading pipeline which:
    - Applies NF4 quantization automatically
    - Patches attention for T4 GPU compatibility
    - Handles dtype selection (float16 for T4)

    Args:
        config: Full pipeline configuration dictionary.

    Returns:
        Tuple of (model, tokenizer).
    """
    from unsloth import FastLanguageModel

    model_config = config.get("model", {})
    model_name = model_config.get("name", "Qwen/Qwen2.5-7B")
    max_seq_length = model_config.get("max_seq_length", 2048)
    load_in_4bit = model_config.get("load_in_4bit", True)

    # Resolve dtype from config string
    dtype_str = model_config.get("dtype", "float16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    logger.info("=" * 60)
    logger.info("[LOAD] Loading model with Unsloth FastLanguageModel")
    logger.info("=" * 60)
    logger.info(f"  Model name       : {model_name}")
    logger.info(f"  Max seq length   : {max_seq_length}")
    logger.info(f"  Dtype            : {dtype_str}")
    logger.info(f"  4-bit quant      : {load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Ensure tokenizer has correct padding configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("[INFO] Set pad_token = eos_token (was None).")

    tokenizer.padding_side = "left"  # Left-padding for decoder-only models
    logger.info("[INFO] Padding side set to 'left' for decoder-only model.")

    # Log model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters : {total_params:,}")
    logger.info(f"  Trainable params : {trainable_params:,} (before LoRA)")

    # Log VRAM after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(f"  VRAM after load  : {allocated:.2f} GB")

    logger.info("[OK] Model and tokenizer loaded successfully.")
    return model, tokenizer


# =============================================================================
# QLoRA Adapter Configuration
# =============================================================================

def apply_lora_adapters(
    model: Any,
    config: Dict[str, Any],
) -> Any:
    """
    Apply QLoRA adapters to the model using Unsloth's optimized PEFT integration.

    Targets all linear projection layers in the transformer:
    - Attention: q_proj, k_proj, v_proj, o_proj
    - MLP: gate_proj, up_proj, down_proj

    Args:
        model: The loaded base model from Unsloth.
        config: Full pipeline configuration dictionary.

    Returns:
        Model with LoRA adapters applied.
    """
    from unsloth import FastLanguageModel

    lora_config = config.get("lora", {})
    r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("lora_alpha", 16)
    lora_dropout = lora_config.get("lora_dropout", 0.0)
    bias = lora_config.get("bias", "none")
    target_modules = lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing = lora_config.get(
        "use_gradient_checkpointing", "unsloth"
    )

    logger.info("=" * 60)
    logger.info("[LORA] Applying QLoRA adapters")
    logger.info("=" * 60)
    logger.info(f"  LoRA rank (r)    : {r}")
    logger.info(f"  LoRA alpha       : {lora_alpha}")
    logger.info(f"  Alpha/r ratio    : {lora_alpha / r:.2f}")
    logger.info(f"  Dropout          : {lora_dropout}")
    logger.info(f"  Bias             : {bias}")
    logger.info(f"  Target modules   : {target_modules}")
    logger.info(f"  Gradient ckpt    : {use_gradient_checkpointing}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=config.get("data", {}).get("seed", 42),
    )

    # Log trainable parameter count after LoRA
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = trainable_params / total_params * 100

    logger.info(f"  Total parameters : {total_params:,}")
    logger.info(f"  Trainable params : {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info("[OK] LoRA adapters applied successfully.")

    return model


# =============================================================================
# ORPO Trainer Setup
# =============================================================================

def create_orpo_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: Dict[str, Any],
) -> Any:
    """
    Create and configure the ORPOTrainer from TRL.

    ORPO combines SFT and preference alignment in a single training step,
    eliminating the need for a reference model. The beta parameter controls
    the weight of the odds-ratio preference loss.

    Args:
        model: Model with LoRA adapters applied.
        tokenizer: The tokenizer.
        train_dataset: Training dataset with (prompt, chosen, rejected) columns.
        val_dataset: Optional validation dataset.
        config: Full pipeline configuration dictionary.

    Returns:
        Configured ORPOTrainer instance.
    """
    from trl import ORPOConfig, ORPOTrainer

    train_config = config.get("training", {})
    wandb_config = config.get("wandb", {})

    # Build ORPOConfig from YAML settings
    orpo_args = ORPOConfig(
        # Output
        output_dir=train_config.get("output_dir", "/kaggle/working/orpo_checkpoints"),

        # Training schedule
        num_train_epochs=train_config.get("num_train_epochs", 1),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),

        # Optimizer
        learning_rate=train_config.get("learning_rate", 5e-6),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        weight_decay=train_config.get("weight_decay", 0.01),
        optim=train_config.get("optim", "adamw_8bit"),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),

        # ORPO-specific
        beta=train_config.get("beta", 0.1),
        max_length=train_config.get("max_seq_length", 2048),
        max_prompt_length=int(train_config.get("max_seq_length", 2048) * 0.5),

        # Precision -- T4 requires fp16, does NOT support bf16
        fp16=train_config.get("fp16", True),
        bf16=train_config.get("bf16", False),

        # Logging
        logging_steps=train_config.get("logging_steps", 10),
        report_to=train_config.get("report_to", "wandb") if wandb_config.get("enabled", False) else "none",
        run_name=wandb_config.get("run_name", "orpo-run"),

        # Evaluation
        eval_strategy=train_config.get("eval_strategy", "steps") if val_dataset else "no",
        eval_steps=train_config.get("eval_steps", 200) if val_dataset else None,

        # Checkpointing
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 2),

        # Misc
        seed=config.get("data", {}).get("seed", 42),
        remove_unused_columns=False,
    )

    # Compute effective batch size for logging
    effective_batch = (
        orpo_args.per_device_train_batch_size
        * orpo_args.gradient_accumulation_steps
    )

    logger.info("=" * 60)
    logger.info("[TRAIN] ORPO Trainer Configuration")
    logger.info("=" * 60)
    logger.info(f"  Output dir       : {orpo_args.output_dir}")
    logger.info(f"  Epochs           : {orpo_args.num_train_epochs}")
    logger.info(f"  Batch size       : {orpo_args.per_device_train_batch_size}")
    logger.info(f"  Grad accum steps : {orpo_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch  : {effective_batch}")
    logger.info(f"  Learning rate    : {orpo_args.learning_rate}")
    logger.info(f"  LR scheduler     : {orpo_args.lr_scheduler_type}")
    logger.info(f"  Warmup ratio     : {orpo_args.warmup_ratio}")
    logger.info(f"  Weight decay     : {orpo_args.weight_decay}")
    logger.info(f"  ORPO beta        : {orpo_args.beta}")
    logger.info(f"  Max length       : {orpo_args.max_length}")
    logger.info(f"  Max prompt len   : {orpo_args.max_prompt_length}")
    logger.info(f"  Precision        : {'fp16' if orpo_args.fp16 else 'fp32'}")
    logger.info(f"  Optimizer        : {orpo_args.optim}")
    logger.info(f"  Report to        : {orpo_args.report_to}")
    logger.info(f"  Train samples    : {len(train_dataset):,}")
    if val_dataset:
        logger.info(f"  Val samples      : {len(val_dataset):,}")
    logger.info("=" * 60)

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("[OK] ORPOTrainer created successfully.")
    return trainer


# =============================================================================
# Model Saving
# =============================================================================

def save_lora_adapters(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> str:
    """
    Save the trained LoRA adapters to disk.

    Args:
        model: The trained model with LoRA adapters.
        tokenizer: The tokenizer.
        config: Full pipeline configuration dictionary.

    Returns:
        Path to the saved LoRA adapters directory.
    """
    saving_config = config.get("saving", {})
    lora_dir = saving_config.get("lora_output_dir", "/kaggle/working/orpo_lora_adapters")

    Path(lora_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"[SAVE] Saving LoRA adapters to: {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    logger.info("[OK] LoRA adapters saved.")

    return lora_dir


def merge_and_save_model(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> str:
    """
    Merge LoRA adapters with the base model and save the full merged model.

    This creates a standalone model that can be used for inference without
    requiring the LoRA adapter files or the base model separately.

    Note: Merging requires significant RAM. On Kaggle, ensure sufficient
    system memory is available.

    Args:
        model: The trained model with LoRA adapters.
        tokenizer: The tokenizer.
        config: Full pipeline configuration dictionary.

    Returns:
        Path to the saved merged model directory.
    """
    from unsloth import FastLanguageModel

    saving_config = config.get("saving", {})
    merged_dir = saving_config.get("merged_output_dir", "/kaggle/working/orpo_merged_model")

    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"[MERGE] Merging LoRA adapters with base model...")
    logger.info(f"[MERGE] Output directory: {merged_dir}")

    # Merge and save in float16 for inference efficiency
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info("[OK] Merged model saved successfully.")
    return merged_dir


def push_to_hub(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> None:
    """
    Push the model and tokenizer to HuggingFace Hub.

    Requires HF_TOKEN to be set as a Kaggle secret or environment variable.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        config: Full pipeline configuration dictionary.
    """
    saving_config = config.get("saving", {})
    hub_model_id = saving_config.get("hub_model_id", "")

    if not hub_model_id:
        logger.info("[INFO] hub_model_id not set. Skipping push to Hub.")
        return

    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("[WARN] HF_TOKEN not found. Cannot push to Hub.")
        return

    logger.info(f"[PUSH] Pushing model to HuggingFace Hub: {hub_model_id}")

    model.push_to_hub_merged(
        hub_model_id,
        tokenizer,
        save_method="merged_16bit",
        token=token,
    )

    logger.info(f"[OK] Model pushed to Hub: {hub_model_id}")


# =============================================================================
# Memory Management
# =============================================================================

def clear_gpu_memory() -> None:
    """Force GPU memory cleanup between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("[OK] GPU memory cleared.")


# =============================================================================
# Training Callbacks
# =============================================================================

def _log_training_metrics(trainer: Any) -> None:
    """
    Log final training metrics after training completes.

    Args:
        trainer: The completed ORPOTrainer instance.
    """
    if not hasattr(trainer, "state") or trainer.state is None:
        return

    log_history = trainer.state.log_history
    if not log_history:
        return

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)

    # Find the last entry with training loss
    train_losses = [
        entry["loss"] for entry in log_history
        if "loss" in entry
    ]
    eval_losses = [
        entry["eval_loss"] for entry in log_history
        if "eval_loss" in entry
    ]

    if train_losses:
        print(f"  Initial train loss : {train_losses[0]:.4f}")
        print(f"  Final train loss   : {train_losses[-1]:.4f}")
        print(f"  Best train loss    : {min(train_losses):.4f}")

    if eval_losses:
        print(f"  Initial eval loss  : {eval_losses[0]:.4f}")
        print(f"  Final eval loss    : {eval_losses[-1]:.4f}")
        print(f"  Best eval loss     : {min(eval_losses):.4f}")

    # ORPO-specific metrics
    orpo_rewards_chosen = [
        entry.get("rewards/chosen") for entry in log_history
        if entry.get("rewards/chosen") is not None
    ]
    orpo_rewards_rejected = [
        entry.get("rewards/rejected") for entry in log_history
        if entry.get("rewards/rejected") is not None
    ]

    if orpo_rewards_chosen and orpo_rewards_rejected:
        print(f"  Final reward (chosen)   : {orpo_rewards_chosen[-1]:.4f}")
        print(f"  Final reward (rejected) : {orpo_rewards_rejected[-1]:.4f}")
        margin = orpo_rewards_chosen[-1] - orpo_rewards_rejected[-1]
        print(f"  Final reward margin     : {margin:.4f}")

    # Training speed
    last_entry = log_history[-1] if log_history else {}
    if "train_runtime" in last_entry:
        runtime = last_entry["train_runtime"]
        samples_per_sec = last_entry.get("train_samples_per_second", 0)
        print(f"  Total runtime      : {runtime:.0f} seconds ({runtime/3600:.1f} hours)")
        print(f"  Throughput         : {samples_per_sec:.2f} samples/sec")

    print("=" * 60 + "\n")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def run_orpo_training(
    config: Dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
) -> Any:
    """
    Execute the full ORPO training pipeline.

    This is the main entry point for training. It:
    1. Loads the model with Unsloth (4-bit)
    2. Applies QLoRA adapters
    3. Creates and runs the ORPOTrainer
    4. Saves LoRA adapters
    5. Optionally merges and saves the full model
    6. Optionally pushes to HuggingFace Hub

    Args:
        config: Full pipeline configuration dictionary.
        train_dataset: Training dataset with (prompt, chosen, rejected) columns.
        val_dataset: Optional validation dataset.

    Returns:
        The completed ORPOTrainer instance.
    """
    logger.info("=" * 60)
    logger.info("[START] ORPO Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load model ──────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)

    # Log VRAM after model load
    if torch.cuda.is_available():
        from utils import print_vram_usage
        print_vram_usage("After model load")

    # ── Step 2: Apply LoRA adapters ─────────────────────────────────────────
    model = apply_lora_adapters(model, config)

    if torch.cuda.is_available():
        from utils import print_vram_usage
        print_vram_usage("After LoRA")

    # ── Step 3: Create trainer ──────────────────────────────────────────────
    trainer = create_orpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    # ── Step 4: Train ───────────────────────────────────────────────────────
    logger.info("[TRAIN] Starting ORPO training...")
    trainer_stats = trainer.train()

    # Log training summary
    _log_training_metrics(trainer)

    if torch.cuda.is_available():
        from utils import print_vram_usage
        print_vram_usage("After training")

    # ── Step 5: Save LoRA adapters ──────────────────────────────────────────
    lora_path = save_lora_adapters(model, tokenizer, config)
    logger.info(f"[SAVE] LoRA adapters at: {lora_path}")

    # ── Step 6: Merge with base model (optional) ───────────────────────────
    saving_config = config.get("saving", {})
    merged_dir = saving_config.get("merged_output_dir", "")

    if merged_dir:
        try:
            merged_path = merge_and_save_model(model, tokenizer, config)
            logger.info(f"[SAVE] Merged model at: {merged_path}")
        except Exception as e:
            logger.warning(f"[WARN] Model merge failed (may need more RAM): {e}")
            logger.info("[INFO] LoRA adapters were saved successfully. "
                       "You can merge manually later.")

    # ── Step 7: Push to Hub (optional) ──────────────────────────────────────
    if saving_config.get("push_to_hub", False):
        try:
            push_to_hub(model, tokenizer, config)
        except Exception as e:
            logger.warning(f"[WARN] Push to Hub failed: {e}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    clear_gpu_memory()

    logger.info("=" * 60)
    logger.info("[DONE] ORPO Training Pipeline Complete")
    logger.info("=" * 60)

    return trainer


# =============================================================================
# Quick Inference Test
# =============================================================================

def test_inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "What is machine learning?",
    max_new_tokens: int = 256,
) -> str:
    """
    Run a quick inference test on the trained model.

    Args:
        model: The trained model (with or without merged LoRA).
        tokenizer: The tokenizer.
        prompt: Test prompt to generate from.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The generated text response.
    """
    from unsloth import FastLanguageModel

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Format prompt in ChatML for Qwen 2.5
    chatml_prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(chatml_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated portion
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("  INFERENCE TEST")
    print("=" * 60)
    print(f"  Prompt: {prompt}")
    print(f"  Response: {response[:500]}")
    print("=" * 60 + "\n")

    return response


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import initialize_pipeline
    from data_prep import prepare_orpo_dataset

    # Default config path
    config_path = "configs/orpo_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # Initialize pipeline (config, seed, GPU check, tokens, dirs)
    config = initialize_pipeline(config_path)

    # Prepare data
    train_ds, val_ds = prepare_orpo_dataset(config)

    # Run training
    trainer = run_orpo_training(config, train_ds, val_ds)

    # Quick inference test
    test_inference(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        prompt="Explain the concept of ORPO in machine learning.",
    )
