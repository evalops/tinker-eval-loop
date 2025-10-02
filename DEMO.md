# End-to-End Demo Walkthrough

This document walks through a complete demo of the evaluation-driven fine-tuning loop.

## What the demo does

1. **Loads 20 QA training examples** from `demo_data.jsonl`
2. **Fine-tunes** Llama-3.1-8B with LoRA for 5 steps (Round 1)
3. **Evaluates** the model on 5 test questions
4. **Checks threshold**: If accuracy < 75%, triggers Round 2
5. **Adjusts hyperparameters**: Reduces learning rate by 40% (0.0003 → 0.00018)
6. **Repeats** up to 3 rounds or until threshold is met

## Expected output

```
╔════════════════════════════════════════════════════════════╗
║   Tinker Evaluation-Driven Fine-Tuning Demo               ║
╚════════════════════════════════════════════════════════════╝

✓ Tinker API key found
📦 Installing dependencies...
✓ Dependencies installed

🚀 Starting evaluation-driven training loop...
   - Base model: meta-llama/Llama-3.1-8B-Instruct
   - Training data: 20 examples (demo_data.jsonl)
   - Max rounds: 3
   - Eval threshold: 0.75
   - Initial LR: 0.0003 (decays by 0.6x per round)

Creating LoRA training client for meta-llama/Llama-3.1-8B-Instruct...
Loaded 20 examples from demo_data.jsonl
Filtered to 20 valid examples
Prepared 20 training datums

=== Training round 1/3 ===
Saving model checkpoint...
Checkpoint saved at tinker://checkpoint-abc123

Running evaluations...
  Running 5 test questions...
    ✗ Question 1: Incorrect
    ✓ Question 2: Correct
    ✗ Question 3: Incorrect
    ✓ Question 4: Correct
    ✗ Question 5: Incorrect
  Evaluation complete: 2/5 correct
  Accuracy: 40.00%
Evaluation score: 0.4000

Score below threshold (0.75). Preparing next round...

=== Training round 2/3 ===
Saving model checkpoint...
Checkpoint saved at tinker://checkpoint-def456

Running evaluations...
  Running 5 test questions...
    ✓ Question 1: Correct
    ✓ Question 2: Correct
    ✓ Question 3: Correct
    ✗ Question 4: Incorrect
    ✓ Question 5: Correct
  Evaluation complete: 4/5 correct
  Accuracy: 80.00%
Evaluation score: 0.8000

Target met: 0.8000 >= 0.75. Stopping.

Training loop completed.

✅ Demo complete!

What happened:
  1. Loaded 20 training examples
  2. Fine-tuned model with LoRA on Tinker infrastructure
  3. Evaluated model on QA tasks
  4. If score < 0.75: adjusted LR and started Round 2
  5. Repeated until threshold met or max rounds reached
```

## Key observations

### Round 1
- **LR**: 0.0003
- **Score**: ~40% (below threshold)
- **Action**: Decay LR to 0.00018, start Round 2

### Round 2
- **LR**: 0.00018 (60% of previous)
- **Score**: ~80% (above threshold)
- **Action**: Stop training ✓

## Customizing the demo

### Use your own data
Replace `demo_data.jsonl` with your JSONL file:

```json
{"instruction": "Your instruction here", "output": "Expected output"}
```

### Adjust the threshold
Edit `demo_config.json`:

```json
{
  "eval_threshold": 0.85,  // Stricter requirement
  "max_rounds": 5          // More rounds allowed
}
```

### Enable EvalOps tracking
Set environment variables and update config:

```bash
export EVALOPS_API_KEY=your-key
```

```json
{
  "evalops_enabled": true,
  "evalops_test_suite_id": "your-suite-id"
}
```

Every round will now be tracked in EvalOps with full metrics and checkpoint URIs.

### Use real Inspect AI tasks
Replace the simple evaluator in `run_evaluations()` with Inspect AI integration:

```python
from inspect_ai import eval
from inspect_ai.dataset import example_dataset

# Run actual Inspect AI tasks
results = await eval(
    tasks=["ifeval", "mmlu"],
    model=model_path,
    model_args={"renderer": renderer_name}
)
```

## Production checklist

Before using in production:

- [ ] Replace `simple_eval.py` with real Inspect AI task integration
- [ ] Implement proper data pipeline (deduplication, quality filters)
- [ ] Add batching and multiple steps per round
- [ ] Enable gradient accumulation and mixed precision
- [ ] Add checkpointing and resume capability
- [ ] Configure EvalOps integration for centralized tracking
- [ ] Set up alerting for threshold violations
- [ ] Add data selection based on failure analysis
- [ ] Tune hyperparameters for your specific domain
