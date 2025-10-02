# Key Improvements Based on Tinker Docs Review

## ✅ Implemented

### 1. **Proper Renderers with Loss Masking**
- **What**: Using `renderers.build_supervised_example()` instead of naive tokenization
- **Why**: Ensures loss is only computed on assistant outputs, not prompts. Aligns with model's chat format.
- **Impact**: Fixes fundamental training correctness
- **Code**: See `data_loader.py` - falls back gracefully if tinker-cookbook unavailable

### 2. **LoRA Rank Configuration**
- **What**: Added `lora_rank` parameter (default: 16)
- **Why**: Controls adapter capacity and training cost
- **Impact**: Allows tuning speed/quality tradeoff
- **Code**: `config_schema.py` + passed to `create_lora_training_client(rank=...)`

### 3. **save_weights_for_sampler() for Evaluation**
- **What**: Using `save_weights_for_sampler()` instead of `save_state()` for eval
- **Why**: `save_state()` includes optimizer state (for resuming). Eval only needs weights.
- **Impact**: Faster checkpointing, correct eval pattern
- **Code**: `trainer_with_eval.py` line ~234

### 4. **Recommended Learning Rate Formula**
- **What**: Tinker's LR formula: `LR = 5e-5 × 10 × (2000/H_m)^P_m`
- **Why**: Tuned for LoRA + Tinker infrastructure across model sizes
- **Impact**: Better convergence, less hyperparameter search
- **Code**: `hyperparam_utils.py` - `get_recommended_lr(model_name)`
- **Usage**: Set `use_recommended_lr: true` in config

### 5. **Warmup + Cosine LR Schedule**
- **What**: Linear warmup → cosine decay to `min_lr`
- **Why**: Stabilizes training, standard best practice
- **Impact**: Reduces instability, improves final metrics
- **Code**: `hyperparam_utils.py` - `get_lr_with_warmup(step, ...)`

## 🚧 To Implement (Production-Ready)

### 6. **Async Futures for Overlapping Requests**
- **What**: Use `forward_backward_async()` and `optim_step_async()` with batching
- **Why**: Overlap compute/network, avoid missing Tinker's ~10s clock cycles
- **Impact**: 2-3x faster training throughput
- **Code**: Replace `run_training_round()` with async micro-batching loop
- **Docs**: https://tinker-docs.thinkingmachines.ai/async

```python
async def run_training_round_async(training_client, datums, cfg, step_offset):
    batches = [datums[i:i+cfg.batch_size] for i in range(0, len(datums), cfg.batch_size)]
    
    futures = []
    for i in range(min(cfg.steps_per_round, len(batches))):
        lr = get_lr_with_warmup(step=step_offset + i, ...)
        fut = await training_client.forward_backward_async(batches[i], "cross_entropy")
        futures.append(fut)
    
    await asyncio.gather(*futures)
    await training_client.optim_step_async(types.AdamParams(learning_rate=lr))
```

### 7. **Proper Inspect AI Integration**
- **What**: Use `InspectAPIFromTinkerSampling` with actual Inspect tasks
- **Why**: Real evals instead of simulated scores
- **Impact**: Demo credibility, production evals
- **Code**: In `run_evaluations()`, create sampling client from weights_uri:

```python
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from inspect_ai import eval_async

sampling_client = service_client.create_sampling_client(model_path=model_path)
api = InspectAPIFromTinkerSampling(
    renderer_name=renderer_name,
    model_name=model_name,
    sampling_client=sampling_client,
)
results = await eval_async(tasks=tasks, model=InspectAIModel(api=api))
```

### 8. **Batch Multiple Examples Per Step**
- **What**: Currently doing 1 step per round. Should do `steps_per_round` with batching.
- **Why**: Realistic training (need 100+ steps minimum per docs)
- **Impact**: Proper gradient signal, better results
- **Dependency**: Requires #6 (async futures)

### 9. **Resume from Checkpoint**
- **What**: `load_state()` to continue from saved checkpoint
- **Why**: Long training runs, experimentation, recovery
- **Code**: Add `--resume` flag + `load_state(checkpoint_uri)` before loop

### 10. **Model-Specific Defaults**
- **What**: Auto-detect renderer from model name, validate compatibility
- **Why**: Prevent silent failures from renderer/model mismatch
- **Code**: Add model→renderer mapping in config validation

## Priority for Next PR

1. **Async futures + batching** (#6 + #8) - Biggest performance/correctness win
2. **Inspect AI integration** (#7) - Makes demo real and production-ready
3. **Resume capability** (#9) - Common production need

## Notes

- Current implementation works but is simplified for demo purposes
- Renderers fallback gracefully if tinker-cookbook unavailable
- LR scheduler optional - can use static LR for simple demos
- All improvements maintain backward compatibility with existing configs
