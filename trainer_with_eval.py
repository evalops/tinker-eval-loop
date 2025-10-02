#!/usr/bin/env python
"""
trainer_with_eval.py

This script implements a simple evaluation‑driven training loop using the
Tinker API.  It reads a JSON configuration file describing the base model,
training data, evaluation tasks and thresholds, and then iteratively fine‑tunes
the model with LoRA until evaluation metrics meet the desired targets or the
maximum number of rounds is reached.

Because Tinker runs training jobs on managed infrastructure, you must have
access to a valid Tinker API key in your environment.  The script uses
blocking calls for simplicity but can easily be extended to exploit Tinker's
asynchronous futures (see the docs for `forward_backward_async` and
`optim_step_async`)
【645635658704514†L231-L259】.

Note: This example is a scaffold.  It leaves out many details, such as data
tokenisation and evaluation implementation, which depend on your specific
pipeline.  Fill in the TODOs with your own logic.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import tinker
    from tinker import types
except ImportError:
    tinker = None

try:
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
    from inspect_ai.model import Model as InspectAIModel
    from inspect_ai.scorer import model_graded_qa
    from inspect_ai.solver import generate
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
except Exception:
    InspectAIModel = None

try:
    from evalops_client import EvalOpsClient
except ImportError:
    EvalOpsClient = None

try:
    from config_schema import TrainingConfig, load_and_validate_config
    from data_loader import DataLoader
    from simple_eval import run_simple_evaluation
    from hyperparam_utils import get_recommended_lr, get_lr_with_warmup
except ImportError:
    TrainingConfig = None
    DataLoader = None
    run_simple_evaluation = None
    get_recommended_lr = None
    get_lr_with_warmup = None


def prepare_training_data(
    train_file: str,
    tokenizer,
    max_seq_length: int = 2048,
    renderer_name: str = "llama3",
    deduplicate: bool = True,
) -> list:
    """Load and convert training data into a list of Tinker Datum objects.

    The Tinker cookbook includes utilities for building supervised examples via
    `renderers.build_supervised_example`.  For a custom dataset, you can parse
    your JSON/JSONL file and turn each example into a Datum with appropriate
    `model_input` and `loss_fn_inputs` fields.  See the Tinker docs for more
    details【645635658704514†L100-L192】.

    Args:
        train_file: Path to the training JSON/JSONL file.
        tokenizer: A tokenizer object obtained from the Tinker training client.
        max_seq_length: Maximum sequence length for tokenization.
        renderer_name: Name of the renderer for proper formatting.
        deduplicate: Whether to deduplicate examples.

    Returns:
        List of `tinker.types.Datum` objects representing the training batch.
    """
    if DataLoader is None:
        print("Warning: DataLoader not available. Returning empty dataset.")
        return []

    loader = DataLoader(max_seq_length=max_seq_length)
    return loader.prepare_training_data(train_file, tokenizer, renderer_name, deduplicate)


async def run_training_round_async(
    training_client,
    datums: list,
    batch_size: int,
    steps_per_round: int,
    base_lr: float,
    step_offset: int = 0,
    warmup_steps: int = 0,
    max_steps: int = 1000,
    min_lr: float = 1e-6,
) -> int:
    """Run one round of training with proper batching and async futures.

    Args:
        training_client: The Tinker training client (LoRA).
        datums: List of all Datum objects.
        batch_size: Number of examples per batch.
        steps_per_round: Number of training steps to run.
        base_lr: Base learning rate.
        step_offset: Global step offset for LR scheduling.
        warmup_steps: Number of warmup steps.
        max_steps: Total steps for cosine decay.
        min_lr: Minimum LR floor.

    Returns:
        Number of steps executed.
    """
    batches = [datums[i:i+batch_size] for i in range(0, len(datums), batch_size)]
    steps_to_run = min(steps_per_round, len(batches))
    
    for step_idx in range(steps_to_run):
        batch = batches[step_idx % len(batches)]
        global_step = step_offset + step_idx
        
        if get_lr_with_warmup and warmup_steps > 0:
            lr = get_lr_with_warmup(global_step, base_lr, warmup_steps, max_steps, min_lr)
        else:
            lr = base_lr
        
        fwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        await fwd_future
        
        optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=lr))
        await optim_future
    
    return steps_to_run


def run_training_round(training_client, datums: list, learning_rate: float) -> None:
    """Legacy sync training round (kept for backward compatibility).

    Args:
        training_client: The Tinker training client (LoRA).
        datums: List of Datum objects representing the training batch.
        learning_rate: Learning rate for the optimizer.
    """
    fwd_result = training_client.forward_backward(datums, loss_fn="cross_entropy")
    training_client.optim_step(types.AdamParams(learning_rate=learning_rate))


async def run_evaluations(
    model_path: str,
    model_name: str,
    tasks: list,
    renderer_name: str,
    threshold: float,
    training_client: Optional[Any] = None,
    evalops_client: Optional[Any] = None,
    test_suite_id: Optional[str] = None,
    round_number: Optional[int] = None,
) -> float:
    """Run evaluation tasks and return an aggregate score.

    This is a placeholder demonstrating how to call evaluations via the
    Inspect AI integration described in the Tinker docs【745100421330604†L122-L185】.  You
    should modify this function to suit your evaluation pipeline.  For example,
    you might call `run_inspect_evals` via `subprocess` or build your own
    `SamplingClientEvaluator`.

    If EvalOps integration is enabled, this function will also submit the
    evaluation results to the EvalOps platform for tracking and analysis.

    Args:
        model_path: The path to the model checkpoint.  For Tinker models, use
            the `tinker://...` syntax as described in the docs.
        model_name: The name of the base model used (e.g., "meta-llama/Llama-3.1-8B").
        tasks: A list of evaluation task identifiers (e.g., "inspect_evals/ifeval").
        renderer_name: The name of the renderer to use for message formatting.
        threshold: A target score; used to decide whether training should continue.
        evalops_client: Optional EvalOps client for submitting results.
        test_suite_id: Optional test suite ID in EvalOps.
        round_number: Optional training round number.

    Returns:
        A float representing the aggregated evaluation score.  Higher is better.
    """
    if run_simple_evaluation is not None:
        score = run_simple_evaluation(
            training_client, model_path, tasks, round_number=round_number or 1
        )
    else:
        score = np.random.rand()
        print(f"  Using simulated score: {score:.4f} (implement real evaluation for production)")

    if evalops_client and test_suite_id:
        try:
            metrics = {
                "aggregate_score": float(score),
                "threshold": threshold,
                "tasks_evaluated": len(tasks),
            }

            result = await evalops_client.submit_training_results(
                test_suite_id=test_suite_id,
                round_number=round_number or 1,
                model_checkpoint=model_path,
                metrics=metrics,
                metadata={
                    "base_model": model_name,
                    "tasks": tasks,
                    "renderer": renderer_name,
                },
            )
            print(f"  Submitted to EvalOps: test run {result['data']['id']}")
        except Exception as e:
            print(f"  Warning: Failed to submit to EvalOps: {e}")

    return score


async def async_main(config_path: str) -> None:
    """Main training loop with async EvalOps integration."""
    if TrainingConfig is None:
        raise ImportError("config_schema module required. Please ensure all dependencies are installed.")

    config = load_and_validate_config(config_path)

    if tinker is None:
        raise ImportError("The `tinker` package is not installed.  Please install it via `pip install tinker`.")

    service_client = tinker.ServiceClient()
    base_model = config.base_model
    max_rounds = config.max_rounds
    eval_threshold = config.eval_threshold
    tasks = config.eval_tasks
    renderer_name = config.renderer_name

    if config.use_recommended_lr and get_recommended_lr:
        learning_rate = get_recommended_lr(base_model)
        print(f"Using recommended LR for {base_model}: {learning_rate:.2e}")
    else:
        learning_rate = config.learning_rate
    
    global_step = 0

    evalops_enabled = config.evalops_enabled
    test_suite_id = config.evalops_test_suite_id
    
    evalops_client = None
    if evalops_enabled:
        if EvalOpsClient is None:
            print("Warning: EvalOps integration requested but evalops_client module not available.")
        elif not test_suite_id:
            print("Warning: evalops_enabled=true but evalops_test_suite_id not specified in config.")
        else:
            try:
                evalops_client = EvalOpsClient()
                print(f"EvalOps integration enabled (test suite: {test_suite_id})")
            except ValueError as e:
                print(f"Warning: Could not initialize EvalOps client: {e}")

    print(f"Creating LoRA training client for {base_model} (rank={config.lora_rank})...")
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=config.lora_rank,
    )

    tokenizer = training_client.get_tokenizer()

    datums = prepare_training_data(
        train_file=config.train_file,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        renderer_name=config.renderer_name,
        deduplicate=True,
    )
    if not datums:
        print("Warning: no training data loaded. Check that your training file has valid examples.")

    try:
        for round_idx in range(1, max_rounds + 1):
            print(f"\n=== Training round {round_idx}/{max_rounds} ===")
            
            if hasattr(training_client, 'forward_backward_async'):
                steps_executed = await run_training_round_async(
                    training_client=training_client,
                    datums=datums,
                    batch_size=config.batch_size,
                    steps_per_round=config.steps_per_round,
                    base_lr=learning_rate,
                    step_offset=global_step,
                    warmup_steps=config.warmup_steps,
                    max_steps=config.max_steps,
                    min_lr=config.min_lr,
                )
                print(f"  Completed {steps_executed} training steps")
                global_step += steps_executed
            else:
                if get_lr_with_warmup and config.warmup_steps > 0:
                    current_lr = get_lr_with_warmup(
                        step=global_step,
                        base_lr=learning_rate,
                        warmup_steps=config.warmup_steps,
                        max_steps=config.max_steps,
                        min_lr=config.min_lr,
                    )
                    print(f"  Step {global_step}: LR = {current_lr:.2e}")
                else:
                    current_lr = learning_rate
                
                run_training_round(training_client, datums, current_lr)
                global_step += config.steps_per_round

            print("Saving model checkpoint...")
            weights_uri = training_client.save_weights_for_sampler(name=f"round_{round_idx}")
            state_uri = weights_uri.result().path if hasattr(weights_uri, 'result') else weights_uri
            print(f"Checkpoint saved at {state_uri}")

            print("Running evaluations...")
            score = await run_evaluations(
                model_path=state_uri,
                model_name=base_model,
                tasks=tasks,
                renderer_name=renderer_name,
                threshold=eval_threshold,
                training_client=training_client,
                evalops_client=evalops_client,
                test_suite_id=test_suite_id,
                round_number=round_idx,
            )
            print(f"Evaluation score: {score:.4f}")

            if score >= eval_threshold:
                print(f"Target met: {score:.4f} >= {eval_threshold}.  Stopping.")
                break
            else:
                print(f"Score below threshold ({eval_threshold}).  Preparing next round...")
                learning_rate *= config.lr_decay

        print("Training loop completed.")
    finally:
        if evalops_client:
            await evalops_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation‑driven fine‑tuning loop")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    asyncio.run(async_main(args.config))


if __name__ == "__main__":
    main()
