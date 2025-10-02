"""
Mock Tinker client for offline demos and testing.

Allows running the evaluation loop without cloud API access.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class MockTokenizer:
    """Mock tokenizer for offline mode."""
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.split()
        return list(range(len(tokens)))
    
    def decode(self, tokens: List[int]) -> str:
        return f"<decoded_{len(tokens)}_tokens>"


class MockFuture:
    """Mock future for sync API."""
    
    def __init__(self, value: Any):
        self.value = value
    
    def result(self):
        return self.value


class MockSaveResult:
    """Mock save result with path."""
    
    def __init__(self, name: str, checkpoint_dir: Path):
        self.path = f"mock://checkpoint/{name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{name}.json"
        checkpoint_file.write_text(json.dumps({"name": name, "mock": True}))


class MockTrainingClient:
    """Mock LoRA training client for offline demos."""
    
    def __init__(self, base_model: str, rank: int, checkpoint_dir: Path):
        self.base_model = base_model
        self.rank = rank
        self.checkpoint_dir = checkpoint_dir
        self.step_count = 0
        self.current_loss = 2.5
    
    def get_tokenizer(self):
        return MockTokenizer()
    
    def forward_backward(self, datums: List[Any], loss_fn: str = "cross_entropy"):
        self.step_count += 1
        self.current_loss *= 0.95
        return MockFuture({"loss": self.current_loss})
    
    async def forward_backward_async(self, datums: List[Any], loss_fn: str = "cross_entropy"):
        await asyncio.sleep(0.01)
        self.step_count += 1
        self.current_loss *= 0.95
        current_loss_value = self.current_loss
        
        class AsyncFuture:
            def __await__(self):
                async def _wait():
                    return {"loss": current_loss_value}
                return _wait().__await__()
        
        return AsyncFuture()
    
    def optim_step(self, params):
        return MockFuture({"success": True})
    
    async def optim_step_async(self, params):
        await asyncio.sleep(0.01)
        
        class AsyncFuture:
            def __await__(self):
                async def _wait():
                    return {"success": True}
                return _wait().__await__()
        
        return AsyncFuture()
    
    def save_weights_for_sampler(self, name: str = "checkpoint"):
        result = MockSaveResult(name, self.checkpoint_dir)
        return MockFuture(result)
    
    def save_state(self, name: str = "checkpoint"):
        result = MockSaveResult(f"{name}_state", self.checkpoint_dir)
        return MockFuture(result)
    
    def load_state(self, path: str):
        print(f"Loaded state from {path}")
        return MockFuture({"success": True})


class MockSamplingClient:
    """Mock sampling client for evaluations."""
    
    def sample(self, prompt, sampling_params, num_samples=1):
        return MockFuture({"sequences": [{"tokens": [1, 2, 3]}]})


class MockServiceClient:
    """Mock Tinker service client."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path("./mock_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def create_lora_training_client(self, base_model: str, rank: int = 16):
        print(f"[MOCK MODE] Creating LoRA training client for {base_model} (rank={rank})")
        return MockTrainingClient(base_model, rank, self.checkpoint_dir)
    
    def create_sampling_client(self, base_model: Optional[str] = None, model_path: Optional[str] = None):
        print(f"[MOCK MODE] Creating sampling client for {model_path or base_model}")
        return MockSamplingClient()
    
    def get_server_capabilities(self):
        class Capabilities:
            supported_models = [
                type('Model', (), {'model_name': 'meta-llama/Llama-3.1-8B-Instruct'}),
                type('Model', (), {'model_name': 'meta-llama/Llama-3.1-70B'}),
            ]
        return Capabilities()


class MockTypes:
    """Mock types module."""
    
    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs
    
    class ModelInput:
        @staticmethod
        def from_ints(tokens):
            return tokens
        
        def to_ints(self):
            return self if isinstance(self, list) else []
    
    class AdamParams:
        def __init__(self, learning_rate: float):
            self.learning_rate = learning_rate
    
    class SamplingParams:
        def __init__(self, max_tokens: int = 100, temperature: float = 0.7, stop: List[str] = None, top_p: float = 1.0):
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.stop = stop or []
            self.top_p = top_p


def create_mock_tinker_module():
    """Create a mock tinker module for offline use."""
    
    class MockTinker:
        ServiceClient = MockServiceClient
        types = MockTypes
    
    return MockTinker
