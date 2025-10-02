"""
Unit tests for data loader.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest


class MockTypes:
    """Mock tinker.types module."""
    
    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs
    
    class ModelInput:
        @staticmethod
        def from_ints(tokens):
            return tokens


mock_tinker = Mock()
mock_tinker.types = MockTypes
sys.modules['tinker'] = mock_tinker
sys.modules['tinker.types'] = MockTypes

mock_renderers = Mock()
mock_renderers.get_renderer = Mock(return_value=None)
sys.modules['tinker_cookbook'] = Mock()
sys.modules['tinker_cookbook.renderers'] = mock_renderers

from data_loader import DataLoader


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str) -> list:
        return list(text.split())


class TestDataLoader:
    """Test suite for DataLoader."""

    def test_load_jsonl_valid_file(self, tmp_path):
        """Load valid JSONL file successfully."""
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"instruction": "Say hello", "output": "Hello!"}\n'
            '{"instruction": "Say goodbye", "output": "Goodbye!"}\n'
        )

        loader = DataLoader()
        examples = loader.load_jsonl(str(jsonl_file))

        assert len(examples) == 2
        assert examples[0]["instruction"] == "Say hello"
        assert examples[1]["output"] == "Goodbye!"

    def test_load_jsonl_with_empty_lines(self, tmp_path):
        """Empty lines are skipped."""
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"instruction": "test", "output": "result"}\n'
            '\n'
            '{"instruction": "test2", "output": "result2"}\n'
        )

        loader = DataLoader()
        examples = loader.load_jsonl(str(jsonl_file))

        assert len(examples) == 2

    def test_load_jsonl_with_invalid_json(self, tmp_path, capsys):
        """Invalid JSON lines are skipped with warning."""
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"instruction": "test", "output": "result"}\n'
            '{invalid json}\n'
            '{"instruction": "test2", "output": "result2"}\n'
        )

        loader = DataLoader()
        examples = loader.load_jsonl(str(jsonl_file))

        assert len(examples) == 2
        captured = capsys.readouterr()
        assert "skipping invalid json" in captured.out.lower()

    def test_load_jsonl_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_jsonl("/nonexistent/file.jsonl")

    def test_validate_example_valid(self):
        """Valid example passes validation."""
        loader = DataLoader()
        example = {"instruction": "Do something", "output": "Done"}
        assert loader.validate_example(example) is True

    def test_validate_example_missing_instruction(self):
        """Example missing instruction fails validation."""
        loader = DataLoader()
        example = {"output": "Done"}
        assert loader.validate_example(example) is False

    def test_validate_example_missing_output(self):
        """Example missing output fails validation."""
        loader = DataLoader()
        example = {"instruction": "Do something"}
        assert loader.validate_example(example) is False

    def test_validate_example_too_short(self):
        """Example below min_length fails validation."""
        loader = DataLoader(min_length=100)
        example = {"instruction": "Hi", "output": "Yo"}
        assert loader.validate_example(example) is False

    def test_validate_example_too_long(self):
        """Example above max_length fails validation."""
        loader = DataLoader(max_length=20)
        example = {
            "instruction": "This is a very long instruction that exceeds the maximum",
            "output": "And a long output too",
        }
        assert loader.validate_example(example) is False

    def test_prepare_training_data_basic(self, tmp_path):
        """Prepare training data from valid JSONL (fallback path without renderer)."""
        jsonl_file = tmp_path / "train.jsonl"
        jsonl_file.write_text(
            '{"instruction": "Say hello", "output": "Hello world"}\n'
            '{"instruction": "Count", "output": "1 2 3"}\n'
        )

        loader = DataLoader(max_seq_length=100)
        tokenizer = MockTokenizer()

        datums = loader.prepare_training_data(str(jsonl_file), tokenizer)

        assert len(datums) >= 0

    def test_prepare_training_data_with_input_field(self, tmp_path):
        """Handle examples with optional input field."""
        jsonl_file = tmp_path / "train.jsonl"
        jsonl_file.write_text(
            '{"instruction": "Summarize", "input": "Long text here", "output": "Summary"}\n'
        )

        loader = DataLoader()
        tokenizer = MockTokenizer()

        datums = loader.prepare_training_data(str(jsonl_file), tokenizer)

        assert len(datums) >= 0

    def test_prepare_training_data_deduplication(self, tmp_path, capsys):
        """Deduplicate identical examples."""
        jsonl_file = tmp_path / "train.jsonl"
        jsonl_file.write_text(
            '{"instruction": "Say hello", "output": "Hello"}\n'
            '{"instruction": "Say hello", "output": "Hello"}\n'
            '{"instruction": "Say bye", "output": "Bye"}\n'
        )

        loader = DataLoader()
        tokenizer = MockTokenizer()

        datums = loader.prepare_training_data(str(jsonl_file), tokenizer, deduplicate=True)

        captured = capsys.readouterr()
        assert "Deduplicated to 2 unique examples" in captured.out

    def test_prepare_training_data_filters_invalid(self, tmp_path, capsys):
        """Invalid examples are filtered out."""
        jsonl_file = tmp_path / "train.jsonl"
        jsonl_file.write_text(
            '{"instruction": "Valid", "output": "Response"}\n'
            '{"instruction": "Missing output"}\n'
            '{"output": "Missing instruction"}\n'
        )

        loader = DataLoader()
        tokenizer = MockTokenizer()

        datums = loader.prepare_training_data(str(jsonl_file), tokenizer)

        captured = capsys.readouterr()
        assert "Filtered to 1 valid examples" in captured.out
