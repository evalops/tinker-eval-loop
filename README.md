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
| `requirements.txt` | Dependencies required to run the script. |

## Quickstart

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Tinker API key** in your environment (see [Tinker docs](https://tinker-docs.thinkingmachines.ai) for how to obtain one). For example:

   ```bash
   export TINKER_API_KEY=sk-...
   ```

3. **Prepare your data** (e.g., instruction/output pairs in JSON or JSONL format) and update `eval_loop_config.json` with the correct paths. Optionally specify evaluation tasks and thresholds.

4. **Run the evaluation‑driven loop:**

   ```bash
   python trainer_with_eval.py --config eval_loop_config.json
   ```

   The script will fine‑tune the specified base model using LoRA, run evaluations, and iteratively improve the model until it meets your quality targets or a maximum number of rounds.

## Extending this project

This is a minimal prototype to demonstrate how to build a useful system on top of Tinker. Future extensions could include:

- **Custom data selection** based on evaluation feedback. For example, automatically mine additional examples from your corpora that match prompts where the model performs poorly.

- **Dynamic hyperparameter tuning** (e.g., adjusting LoRA rank or learning rate) using heuristics or Bayesian optimisation.

- **Feedback integration** from users or human graders to generate reward signals for RL fine‑tuning (Tinker supports PPO and importance‑sampling losses).

- **Integration with EvalOps**, using your existing evaluation pipelines to drive the fine‑tuning loop.

## Disclaimer

This code does not run training jobs by itself; it serves as a scaffold. You'll need an active Tinker API key and appropriate computing quotas to execute the training and evaluation steps. Modify the script to fit your particular needs and model lineup. The hyperparameters and thresholds in the sample config are placeholders and should be adjusted based on your use case and dataset size.
