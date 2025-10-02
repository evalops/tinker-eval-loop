"""
Configuration schema validation using Pydantic.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingConfig(BaseModel):
    """Training configuration schema with validation."""

    base_model: str = Field(
        ...,
        description="Base model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct')",
        min_length=1,
    )
    train_file: str = Field(
        ..., description="Path to training data (JSONL format)", min_length=1
    )
    eval_tasks: List[str] = Field(
        default_factory=list,
        description="List of Inspect AI evaluation tasks",
    )
    renderer_name: str = Field(
        default="default", description="Message renderer name for the model"
    )
    learning_rate: float = Field(
        default=1e-4, gt=0.0, le=1.0, description="Initial learning rate"
    )
    eval_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum aggregate score to stop training",
    )
    max_rounds: int = Field(
        default=3, ge=1, le=100, description="Maximum training rounds"
    )
    lr_decay: float = Field(
        default=0.8, gt=0.0, le=1.0, description="Learning rate decay factor per round"
    )
    
    evalops_enabled: bool = Field(
        default=False, description="Enable EvalOps integration"
    )
    evalops_test_suite_id: Optional[str] = Field(
        default=None, description="EvalOps test suite ID for tracking"
    )
    evalops_api_url: Optional[str] = Field(
        default=None, description="EvalOps API URL (defaults to env var or public API)"
    )

    steps_per_round: int = Field(
        default=1, ge=1, description="Training steps per round"
    )
    batch_size: int = Field(
        default=8, ge=1, le=512, description="Training batch size"
    )
    max_seq_length: int = Field(
        default=2048, ge=128, le=32768, description="Maximum sequence length"
    )
    lora_rank: int = Field(
        default=16, ge=1, le=256, description="LoRA rank (adapter dimension)"
    )
    warmup_steps: int = Field(
        default=100, ge=0, description="Learning rate warmup steps"
    )
    max_steps: int = Field(
        default=1000, ge=1, description="Total training steps across all rounds"
    )
    min_lr: float = Field(
        default=1e-6, gt=0.0, description="Minimum learning rate floor"
    )
    use_recommended_lr: bool = Field(
        default=False, description="Use Tinker's recommended LR formula instead of manual LR"
    )
    
    @field_validator("train_file")
    @classmethod
    def validate_train_file_exists(cls, v: str) -> str:
        """Validate that training file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Training file not found: {v}")
        if not path.is_file():
            raise ValueError(f"Training file path is not a file: {v}")
        return v

    @model_validator(mode="after")
    def validate_evalops_config(self) -> "TrainingConfig":
        """Validate EvalOps configuration consistency."""
        if self.evalops_enabled and not self.evalops_test_suite_id:
            raise ValueError(
                "evalops_test_suite_id must be provided when evalops_enabled is true"
            )
        return self


def load_and_validate_config(config_path: str) -> TrainingConfig:
    """
    Load and validate configuration from JSON file.

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        Validated TrainingConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    import json

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        raw_config = json.load(f)

    return TrainingConfig.model_validate(raw_config)
