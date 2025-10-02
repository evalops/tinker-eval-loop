"""
Hyperparameter utilities including recommended learning rate schedules.

Based on Tinker's recommended LR formula:
LR(m) = lr_base × M_LoRA × (2000/H_m)^P_m

Reference: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams
"""

import math
from typing import Dict


MODEL_HIDDEN_SIZES: Dict[str, int] = {
    "meta-llama/Llama-3.1-8B": 4096,
    "meta-llama/Llama-3.1-8B-Instruct": 4096,
    "meta-llama/Llama-3.1-70B": 8192,
    "meta-llama/Llama-3.3-70B-Instruct": 8192,
    "meta-llama/Llama-3.2-1B": 2048,
    "meta-llama/Llama-3.2-3B": 3072,
    "Qwen/Qwen3-8B": 4096,
    "Qwen/Qwen3-8B-Base": 4096,
    "Qwen/Qwen3-30B-A3B": 3584,
    "Qwen/Qwen3-30B-A3B-Base": 3584,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 3584,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 8192,
}


def get_recommended_lr(
    model_name: str,
    lr_base: float = 5e-5,
    lora_multiplier: float = 10.0,
) -> float:
    """
    Get recommended learning rate for a model using Tinker's formula.

    Formula: LR(m) = lr_base × M_LoRA × (2000/H_m)^P_m
    where:
    - lr_base: Base learning rate (default 5e-5)
    - M_LoRA: LoRA multiplier (default 10)
    - H_m: Hidden size of model m
    - P_m: Model-specific exponent (0.0775 for Qwen, 0.781 for Llama)

    Args:
        model_name: Full model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        lr_base: Base learning rate
        lora_multiplier: LoRA multiplier

    Returns:
        Recommended learning rate for the model.
    """
    hidden_size = MODEL_HIDDEN_SIZES.get(model_name)
    if hidden_size is None:
        print(f"Warning: Unknown model {model_name}, using default LR")
        return lr_base * lora_multiplier

    if "llama" in model_name.lower():
        exponent = 0.781
    elif "qwen" in model_name.lower():
        exponent = 0.0775
    else:
        exponent = 0.4

    lr = lr_base * lora_multiplier * math.pow(2000 / hidden_size, exponent)
    return lr


def get_lr_with_warmup(
    step: int,
    base_lr: float,
    warmup_steps: int = 100,
    max_steps: int = 1000,
    min_lr: float = 1e-6,
) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step (0-indexed).
        base_lr: Peak learning rate after warmup.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        min_lr: Minimum learning rate floor.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    
    return max(lr, min_lr)
