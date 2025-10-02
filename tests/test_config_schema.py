"""
Unit tests for configuration schema validation.
"""


import pytest
from pydantic import ValidationError

from config_schema import TrainingConfig, load_and_validate_config


class TestTrainingConfig:
    """Test suite for TrainingConfig validation."""

    def test_valid_minimal_config(self, tmp_path):
        """Valid minimal config passes validation."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config = TrainingConfig(
            base_model="meta-llama/Llama-3.1-8B",
            train_file=str(train_file),
        )

        assert config.base_model == "meta-llama/Llama-3.1-8B"
        assert config.learning_rate == 1e-4
        assert config.max_rounds == 3
        assert config.evalops_enabled is False

    def test_valid_full_config(self, tmp_path):
        """Valid full config with all fields passes validation."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config = TrainingConfig(
            base_model="meta-llama/Llama-3.1-8B",
            train_file=str(train_file),
            eval_tasks=["inspect_evals/mmlu"],
            learning_rate=0.0002,
            eval_threshold=0.9,
            max_rounds=5,
            lr_decay=0.75,
            evalops_enabled=True,
            evalops_test_suite_id="suite-123",
            steps_per_round=10,
            batch_size=16,
            max_seq_length=4096,
        )

        assert config.eval_threshold == 0.9
        assert config.max_rounds == 5
        assert config.evalops_test_suite_id == "suite-123"

    def test_missing_required_fields(self):
        """Missing required fields raises validation error."""
        with pytest.raises(ValidationError, match="base_model"):
            TrainingConfig()

    def test_train_file_not_exists(self):
        """Non-existent training file raises validation error."""
        with pytest.raises(ValidationError, match="not found"):
            TrainingConfig(
                base_model="meta-llama/Llama-3.1-8B",
                train_file="/nonexistent/file.jsonl",
            )

    def test_invalid_learning_rate(self, tmp_path):
        """Invalid learning rate raises validation error."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        with pytest.raises(ValidationError, match="learning_rate"):
            TrainingConfig(
                base_model="meta-llama/Llama-3.1-8B",
                train_file=str(train_file),
                learning_rate=-0.01,
            )

    def test_invalid_eval_threshold(self, tmp_path):
        """Eval threshold outside [0, 1] raises validation error."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        with pytest.raises(ValidationError, match="eval_threshold"):
            TrainingConfig(
                base_model="meta-llama/Llama-3.1-8B",
                train_file=str(train_file),
                eval_threshold=1.5,
            )

    def test_invalid_max_rounds(self, tmp_path):
        """Invalid max_rounds raises validation error."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        with pytest.raises(ValidationError, match="max_rounds"):
            TrainingConfig(
                base_model="meta-llama/Llama-3.1-8B",
                train_file=str(train_file),
                max_rounds=0,
            )

    def test_evalops_enabled_without_test_suite_id(self, tmp_path):
        """EvalOps enabled without test suite ID raises validation error."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        with pytest.raises(ValidationError, match="evalops_test_suite_id"):
            TrainingConfig(
                base_model="meta-llama/Llama-3.1-8B",
                train_file=str(train_file),
                evalops_enabled=True,
            )


class TestLoadAndValidateConfig:
    """Test suite for config file loading."""

    def test_load_valid_config_file(self, tmp_path):
        """Load and validate a valid config file."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config_file = tmp_path / "config.json"
        config_file.write_text(
            f'{{"base_model": "llama-8b", "train_file": "{train_file}"}}'
        )

        config = load_and_validate_config(str(config_file))
        assert config.base_model == "llama-8b"

    def test_config_file_not_found(self):
        """Non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_and_validate_config("/nonexistent/config.json")

    def test_invalid_json_raises_error(self, tmp_path):
        """Invalid JSON in config file raises error."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("{invalid json")

        with pytest.raises(Exception):
            load_and_validate_config(str(config_file))
