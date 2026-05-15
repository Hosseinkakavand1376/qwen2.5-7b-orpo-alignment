"""
data_prep.py -- Data engineering for the ORPO alignment pipeline.

Handles:
  1. Loading SlimOrca (SFT, ShareGPT format) and orpo-dpo-mix-40k (preference pairs)
  2. ChatML template formatting for Qwen 2.5
  3. Unifying both sources into the (prompt, chosen, rejected) schema
  4. Train/validation split
  5. Saving processed datasets to disk

Usage:
    from data_prep import prepare_orpo_dataset
    train_ds, val_ds = prepare_orpo_dataset(config)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# =============================================================================
# ChatML Template for Qwen 2.5
# =============================================================================
# Qwen 2.5 uses the ChatML format with <|im_start|> and <|im_end|> tokens.
#
# Format:
#   <|im_start|>system
#   {system_message}<|im_end|>
#   <|im_start|>user
#   {user_message}<|im_end|>
#   <|im_start|>assistant
#   {assistant_message}<|im_end|>

CHATML_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHATML_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


# =============================================================================
# ChatML Formatting Helpers
# =============================================================================

def format_chatml_message(role: str, content: str) -> str:
    """
    Format a single message in ChatML format.

    Args:
        role: One of 'system', 'user', 'assistant'.
        content: The message text.

    Returns:
        ChatML-formatted string for this message.
    """
    templates = {
        "system": CHATML_SYSTEM,
        "user": CHATML_USER,
        "assistant": CHATML_ASSISTANT,
    }
    template = templates.get(role)
    if template is None:
        raise ValueError(f"Unknown role: {role}. Expected 'system', 'user', or 'assistant'.")
    return template.format(content=content.strip())


def format_chatml_prompt(
    system_message: str,
    user_message: str,
) -> str:
    """
    Build the prompt portion (system + user) in ChatML format.
    Does NOT include the assistant response -- that goes into chosen/rejected.

    Args:
        system_message: The system instruction.
        user_message: The user's query/instruction.

    Returns:
        ChatML-formatted prompt string.
    """
    prompt = ""
    if system_message:
        prompt += format_chatml_message("system", system_message)
    prompt += format_chatml_message("user", user_message)
    return prompt


def format_chatml_response(content: str) -> str:
    """
    Format an assistant response in ChatML format.

    Args:
        content: The assistant's response text.

    Returns:
        ChatML-formatted assistant response string.
    """
    return format_chatml_message("assistant", content)


def messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of message dicts to a full ChatML string.

    Args:
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        Full ChatML-formatted conversation string.
    """
    result = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        result += format_chatml_message(role, content)
    return result


# =============================================================================
# SlimOrca Processing (ShareGPT format --> ChatML)
# =============================================================================

# Mapping from ShareGPT roles to ChatML roles
SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def _parse_slimorca_conversation(conversations: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Parse a single SlimOrca conversation (ShareGPT format) into
    a structured dict with system_message, user_message, assistant_response.

    Args:
        conversations: List of dicts with 'from' and 'value' keys.

    Returns:
        Dict with keys: 'system_message', 'user_message', 'assistant_response'.
        Returns None-valued keys if parsing fails.
    """
    system_message = DEFAULT_SYSTEM_MESSAGE
    user_message = ""
    assistant_response = ""

    for turn in conversations:
        role = turn.get("from", "").lower()
        value = turn.get("value", "").strip()

        if role == "system":
            system_message = value if value else DEFAULT_SYSTEM_MESSAGE
        elif role == "human":
            user_message = value
        elif role == "gpt":
            assistant_response = value

    return {
        "system_message": system_message,
        "user_message": user_message,
        "assistant_response": assistant_response,
    }


def process_slimorca_for_orpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single SlimOrca example to the ORPO-compatible format.

    Since SlimOrca only has instruction-response pairs (no rejected responses),
    these entries provide the SFT signal within ORPO. The rejected field is
    set to an empty string -- the ORPOTrainer handles the SFT-only entries
    by using the chosen response for supervised learning.

    Note: In practice, we will merge these with the preference dataset entries.
    SlimOrca entries where rejected is empty will be filtered or handled
    depending on the training strategy.

    Args:
        example: A single SlimOrca dataset row with 'conversations' key.

    Returns:
        Dict with 'prompt', 'chosen', 'rejected' keys.
    """
    conversations = example.get("conversations", [])
    parsed = _parse_slimorca_conversation(conversations)

    # Build ChatML prompt (system + user)
    prompt = format_chatml_prompt(
        system_message=parsed["system_message"],
        user_message=parsed["user_message"],
    )

    # The SlimOrca response becomes the "chosen" response
    chosen = format_chatml_response(parsed["assistant_response"])

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": "",  # No rejected response in SlimOrca
        "source": "slimorca",
    }


# =============================================================================
# orpo-dpo-mix-40k Processing (already preference-paired)
# =============================================================================

def _extract_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Extract the prompt portion (system + user messages) from a list of
    conversation messages. Stops before the assistant response.

    Args:
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        ChatML-formatted prompt string.
    """
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            break  # Stop before assistant response
        prompt_parts.append(format_chatml_message(role, content))
    return "".join(prompt_parts)


def _extract_response_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Extract the assistant response from a list of conversation messages.

    Args:
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        ChatML-formatted assistant response string.
    """
    for msg in messages:
        if msg.get("role", "") == "assistant":
            return format_chatml_response(msg.get("content", ""))
    return ""


def process_preference_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single orpo-dpo-mix-40k example to the ORPO-compatible format.

    The dataset has 'chosen' and 'rejected' fields, each containing a list
    of message dicts in conversational format.

    Args:
        example: A single dataset row with 'chosen' and 'rejected' keys
                 (each a list of message dicts).

    Returns:
        Dict with 'prompt', 'chosen', 'rejected' keys in ChatML format.
    """
    chosen_messages = example.get("chosen", [])
    rejected_messages = example.get("rejected", [])

    # Extract prompt from the chosen conversation (should be same in rejected)
    prompt = _extract_prompt_from_messages(chosen_messages)

    # Extract assistant responses
    chosen_response = _extract_response_from_messages(chosen_messages)
    rejected_response = _extract_response_from_messages(rejected_messages)

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "source": example.get("source", "orpo-dpo-mix"),
    }


# =============================================================================
# Dataset Loading and Processing
# =============================================================================

def load_slimorca(
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load and process the SlimOrca dataset.

    Args:
        subset_size: Number of samples to use. None = use all.
        seed: Random seed for shuffling/sampling.

    Returns:
        Processed HuggingFace Dataset with 'prompt', 'chosen', 'rejected' columns.
    """
    logger.info("[LOAD] Loading SlimOrca dataset from HuggingFace...")
    ds = load_dataset("Open-Orca/SlimOrca", split="train")
    logger.info(f"[LOAD] SlimOrca loaded: {len(ds)} total samples.")

    # Subsample if requested
    if subset_size is not None and subset_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(subset_size))
        logger.info(f"[LOAD] SlimOrca subsampled to {len(ds)} samples.")

    # Process to ORPO format
    logger.info("[PROC] Processing SlimOrca to ChatML format...")
    ds = ds.map(
        process_slimorca_for_orpo,
        remove_columns=ds.column_names,
        desc="Formatting SlimOrca (ChatML)",
    )

    # Filter out entries with empty user messages or responses
    initial_count = len(ds)
    ds = ds.filter(
        lambda x: len(x["prompt"].strip()) > 0 and len(x["chosen"].strip()) > 0,
        desc="Filtering empty entries",
    )
    filtered_count = initial_count - len(ds)
    if filtered_count > 0:
        logger.info(f"[FILTER] Removed {filtered_count} empty entries from SlimOrca.")

    logger.info(f"[OK] SlimOrca processed: {len(ds)} samples ready.")
    return ds


def load_preference_data(
    dataset_name: str = "mlabonne/orpo-dpo-mix-40k",
    subset_size: Optional[int] = None,
    seed: int = 42,
    remove_toxic: bool = True,
) -> Dataset:
    """
    Load and process the preference dataset (orpo-dpo-mix-40k).

    Args:
        dataset_name: HuggingFace dataset identifier.
        subset_size: Number of preference pairs to sample. None = use all.
        seed: Random seed for shuffling/sampling.
        remove_toxic: If True, filter out toxic-dpo-v0.2 samples.

    Returns:
        Processed HuggingFace Dataset with 'prompt', 'chosen', 'rejected' columns.
    """
    logger.info(f"[LOAD] Loading preference dataset: {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    logger.info(f"[LOAD] Preference dataset loaded: {len(ds)} total samples.")

    # Remove toxic samples if requested
    if remove_toxic and "source" in ds.column_names:
        pre_filter = len(ds)
        ds = ds.filter(
            lambda r: r["source"] != "toxic-dpo-v0.2",
            desc="Removing toxic samples",
        )
        removed = pre_filter - len(ds)
        if removed > 0:
            logger.info(f"[FILTER] Removed {removed} toxic-dpo-v0.2 samples.")

    # Subsample if requested
    if subset_size is not None and subset_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(subset_size))
        logger.info(f"[LOAD] Preference data subsampled to {len(ds)} samples.")

    # Process to ORPO format with ChatML
    logger.info("[PROC] Processing preference data to ChatML format...")
    ds = ds.map(
        process_preference_example,
        remove_columns=ds.column_names,
        desc="Formatting preference pairs (ChatML)",
    )

    # Filter out entries with empty prompt, chosen, or rejected
    initial_count = len(ds)
    ds = ds.filter(
        lambda x: (
            len(x["prompt"].strip()) > 0
            and len(x["chosen"].strip()) > 0
            and len(x["rejected"].strip()) > 0
        ),
        desc="Filtering incomplete pairs",
    )
    filtered_count = initial_count - len(ds)
    if filtered_count > 0:
        logger.info(f"[FILTER] Removed {filtered_count} incomplete preference pairs.")

    logger.info(f"[OK] Preference data processed: {len(ds)} pairs ready.")
    return ds


# =============================================================================
# Dataset Merging and Splitting
# =============================================================================

def merge_datasets(
    sft_dataset: Dataset,
    preference_dataset: Dataset,
) -> Dataset:
    """
    Merge SFT (SlimOrca) and preference datasets into a single dataset.

    The SlimOrca entries have empty 'rejected' fields -- these serve as
    additional SFT signal within the ORPO framework. The preference entries
    have both 'chosen' and 'rejected' for the odds-ratio loss.

    Args:
        sft_dataset: Processed SlimOrca dataset.
        preference_dataset: Processed preference dataset.

    Returns:
        Merged HuggingFace Dataset.
    """
    from datasets import concatenate_datasets

    logger.info(
        f"[MERGE] Merging datasets: {len(sft_dataset)} SFT + "
        f"{len(preference_dataset)} preference = "
        f"{len(sft_dataset) + len(preference_dataset)} total."
    )

    # Ensure both datasets have the same columns
    expected_columns = {"prompt", "chosen", "rejected", "source"}
    for ds_name, ds in [("SFT", sft_dataset), ("Preference", preference_dataset)]:
        missing = expected_columns - set(ds.column_names)
        if missing:
            raise ValueError(
                f"{ds_name} dataset is missing columns: {missing}. "
                f"Found: {ds.column_names}"
            )

    merged = concatenate_datasets([sft_dataset, preference_dataset])
    merged = merged.shuffle(seed=42)

    logger.info(f"[OK] Merged dataset: {len(merged)} total samples.")
    return merged


def split_dataset(
    dataset: Dataset,
    val_split: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: The full dataset to split.
        val_split: Fraction of data for validation (default 5%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]

    logger.info(
        f"[SPLIT] Train: {len(train_ds)} samples, "
        f"Validation: {len(val_ds)} samples "
        f"(split ratio: {val_split:.1%})."
    )

    return train_ds, val_ds


# =============================================================================
# Dataset Statistics
# =============================================================================

def print_dataset_stats(dataset: Dataset, name: str = "Dataset") -> None:
    """
    Print summary statistics for a processed dataset.

    Args:
        dataset: The HuggingFace Dataset to analyze.
        name: Label for the printout.
    """
    total = len(dataset)
    if total == 0:
        print(f"\n  {name}: EMPTY (0 samples)")
        return

    # Count entries with/without rejected responses
    has_rejected = sum(1 for x in dataset if x["rejected"].strip())
    sft_only = total - has_rejected

    # Compute average lengths
    prompt_lengths = [len(x["prompt"]) for x in dataset]
    chosen_lengths = [len(x["chosen"]) for x in dataset]

    avg_prompt_len = sum(prompt_lengths) / total
    avg_chosen_len = sum(chosen_lengths) / total
    max_prompt_len = max(prompt_lengths)
    max_chosen_len = max(chosen_lengths)

    # Source distribution
    sources = {}
    for x in dataset:
        src = x.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\n" + "=" * 60)
    print(f"  {name} STATISTICS")
    print("=" * 60)
    print(f"  Total samples        : {total:,}")
    print(f"  Preference pairs     : {has_rejected:,}")
    print(f"  SFT-only entries     : {sft_only:,}")
    print(f"  Avg prompt length    : {avg_prompt_len:.0f} chars")
    print(f"  Avg chosen length    : {avg_chosen_len:.0f} chars")
    print(f"  Max prompt length    : {max_prompt_len:,} chars")
    print(f"  Max chosen length    : {max_chosen_len:,} chars")
    print("  -" * 30)
    print("  Source distribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {src:30s} : {count:>6,} ({pct:5.1f}%)")
    print("=" * 60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def prepare_orpo_dataset(
    config: Dict[str, Any],
) -> Tuple[Dataset, Dataset]:
    """
    Full data preparation pipeline: load, process, merge, split.

    This is the main entry point for data preparation.

    Args:
        config: Pipeline config dict (from orpo_config.yaml).

    Returns:
        Tuple of (train_dataset, val_dataset) ready for ORPOTrainer.
    """
    data_config = config.get("data", {})

    sft_dataset_name = data_config.get("sft_dataset", "Open-Orca/SlimOrca")
    sft_subset_size = data_config.get("sft_subset_size", 50000)
    pref_dataset_name = data_config.get("preference_dataset", "mlabonne/orpo-dpo-mix-40k")
    pref_subset_size = data_config.get("preference_subset_size", 13000)
    val_split = data_config.get("val_split", 0.05)
    seed = data_config.get("seed", 42)
    output_dir = data_config.get("output_dir", "/kaggle/working/processed_data")

    logger.info("=" * 60)
    logger.info("[START] ORPO Data Preparation Pipeline")
    logger.info("=" * 60)
    logger.info(f"  SFT source         : {sft_dataset_name}")
    logger.info(f"  SFT subset size    : {sft_subset_size:,}")
    logger.info(f"  Preference source  : {pref_dataset_name}")
    logger.info(f"  Preference subset  : {pref_subset_size:,}")
    logger.info(f"  Val split          : {val_split:.1%}")
    logger.info(f"  Seed               : {seed}")
    logger.info("=" * 60)

    # 1. Load and process SlimOrca (SFT)
    sft_ds = load_slimorca(
        subset_size=sft_subset_size,
        seed=seed,
    )

    # 2. Load and process preference data
    pref_ds = load_preference_data(
        dataset_name=pref_dataset_name,
        subset_size=pref_subset_size,
        seed=seed,
        remove_toxic=True,
    )

    # 3. Print stats for each source
    print_dataset_stats(sft_ds, "SlimOrca (SFT)")
    print_dataset_stats(pref_ds, "Preference Pairs")

    # 4. Merge datasets
    merged_ds = merge_datasets(sft_ds, pref_ds)

    # 5. Split into train/val
    train_ds, val_ds = split_dataset(merged_ds, val_split=val_split, seed=seed)

    # 6. Print final stats
    print_dataset_stats(train_ds, "Final Training Set")
    print_dataset_stats(val_ds, "Final Validation Set")

    # 7. Save to disk
    if output_dir:
        from pathlib import Path
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        train_ds.save_to_disk(str(save_path / "train"))
        val_ds.save_to_disk(str(save_path / "val"))
        logger.info(f"[SAVE] Datasets saved to: {save_path}")

    logger.info("[DONE] Data preparation complete.")
    return train_ds, val_ds


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import load_config, setup_logging, set_seed

    setup_logging()

    # Default config path
    config_path = "configs/orpo_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    set_seed(config.get("data", {}).get("seed", 42))

    train_ds, val_ds = prepare_orpo_dataset(config)

    # Quick sanity check: print a sample
    print("\n--- Sample from training set ---")
    sample = train_ds[0]
    print(f"Prompt:\n{sample['prompt'][:300]}...")
    print(f"\nChosen:\n{sample['chosen'][:300]}...")
    print(f"\nRejected:\n{sample['rejected'][:200] if sample['rejected'] else '(empty -- SFT only)'}...")
    print(f"\nSource: {sample['source']}")
