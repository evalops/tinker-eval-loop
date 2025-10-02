# Evaluation‑Driven LoRA Fine‑Tuning with Tinker

This project implements a proof‑of‑concept evaluation‑driven fine‑tuning loop on top of [Tinker](https://tinker-docs.thinkingmachines.ai). The goal is to continuously improve a model by training it using LoRA and then measuring its performance on a suite of evaluation tasks. When the model fails to meet a specified threshold, the loop collects additional data or modifies hyperparameters and launches a new fine‑tuning job.

## Why evaluation‑driven fine‑tuning?

[Tinker](https://tinker-docs.thinkingmachines.ai) is a low‑level API for LoRA fine‑tuning that offloads distributed training to managed infrastructure. It also provides an evaluation API that can run inline or offline evaluations and integrate with the Inspect AI library. These features make it possible to build a higher‑level service that:

- Automatically benchmarks models using standard tasks or custom tasks.
- Identifies weakness patterns, such as specific topics or reasoning styles, from evaluation results.
- Launches targeted fine‑tuning jobs with new data or tuned hyperparameters when metrics fall below thresholds.
- Tracks progress over multiple rounds of fine‑tuning to quantify the impact of each intervention.

Such a loop can be particularly useful for domains where quality requirements are high and failure modes are diverse (e.g., legal drafting, safety moderation, tutoring).

## Usage overview

This project contains two main components:

| File | Description |
|------|-------------|
| `trainer_with_eval.py` | The main script that orchestrates training and evaluation. It connects to Tinker, creates a LoRA training client, runs fine‑tuning, performs evaluations via Inspect AI, and decides whether to launch further training rounds. |
| `eval_loop_config.json` | A sample configuration file specifying the base model, dataset paths, evaluation tasks, thresholds and hyperparameters. |
| `evalops_client.py` | Python SDK for submitting evaluation results to EvalOps platform. |
| `config_schema.py` | Pydantic schema for configuration validation. |
| `data_loader.py` | JSONL data loader with validation, deduplication, and tokenization. |
| `data_selector.py` | Utilities for mining hard examples based on evaluation failures. |
| `requirements.txt` | Dependencies required to run the script. |
| `tests/` | Unit and integration tests for all components. |

## Quickstart

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Tinker API key** in your environment (see [Tinker docs](https://tinker-docs.thinkingmachines.ai) for how to obtain one). For example:

   ```bash
   export TINKER_API_KEY=sk-...
   ```

3. **(Optional) Configure EvalOps integration** to automatically track all evaluation results:

   ```bash
   export EVALOPS_API_KEY=your-evalops-api-key
   export EVALOPS_API_URL=https://api.evalops.dev  # or your self-hosted instance
   ```

   Then update `eval_loop_config.json` to enable EvalOps:

   ```json
   {
     ...
     "evalops_enabled": true,
     "evalops_test_suite_id": "your-test-suite-id"
   }
   ```

4. **Prepare your data** (e.g., instruction/output pairs in JSON or JSONL format) and update `eval_loop_config.json` with the correct paths. Optionally specify evaluation tasks and thresholds.

5. **Run the evaluation‑driven loop:**

   ```bash
   python trainer_with_eval.py --config eval_loop_config.json
   ```

   The script will fine‑tune the specified base model using LoRA, run evaluations, and iteratively improve the model until it meets your quality targets or a maximum number of rounds. If EvalOps integration is enabled, each evaluation round will be automatically submitted to your EvalOps workspace for tracking and analysis.

## EvalOps Integration

This project includes built-in integration with [EvalOps](https://evalops.dev) to automatically track evaluation results across training rounds. The `evalops_client.py` module provides a lightweight Python SDK that:

- Submits each training round's evaluation results as a test run in EvalOps
- Tracks metrics, model checkpoints, and metadata for each iteration
- Enables centralized monitoring and comparison of fine-tuning experiments

To use EvalOps integration:

1. Create a test suite in your EvalOps workspace
2. Set the `evalops_enabled: true` and `evalops_test_suite_id` in your config
3. Provide your EvalOps API key via the `EVALOPS_API_KEY` environment variable

The client will automatically submit results after each evaluation round, making it easy to track progress over time and compare different fine-tuning runs.

## Extending this project

This is a minimal prototype to demonstrate how to build a useful system on top of Tinker. Future extensions could include:

- **Custom data selection** based on evaluation feedback. For example, automatically mine additional examples from your corpora that match prompts where the model performs poorly.

- **Dynamic hyperparameter tuning** (e.g., adjusting LoRA rank or learning rate) using heuristics or Bayesian optimisation.

- **Feedback integration** from users or human graders to generate reward signals for RL fine‑tuning (Tinker supports PPO and importance‑sampling losses).

- **Advanced EvalOps features**, such as quality gates, automated alerts when metrics drop below thresholds, or integration with regression testing schedules.

## Disclaimer

This code does not run training jobs by itself; it serves as a scaffold. You'll need an active Tinker API key and appropriate computing quotas to execute the training and evaluation steps. Modify the script to fit your particular needs and model lineup. The hyperparameters and thresholds in the sample config are placeholders and should be adjusted based on your use case and dataset size.
