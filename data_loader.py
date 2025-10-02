"""
Data loading utilities for preparing training data from JSONL files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tinker import types
except ImportError:
    types = None


class DataLoader:
    """Load and prepare training data from JSONL files."""

    def __init__(
        self,
        max_seq_length: int = 2048,
        min_length: int = 10,
        max_length: int = 4096,
    ):
        """
        Initialize data loader.

        Args:
            max_seq_length: Maximum sequence length for tokenization.
            min_length: Minimum text length to include.
            max_length: Maximum text length to include.
        """
        self.max_seq_length = max_seq_length
        self.min_length = min_length
        self.max_length = max_length

    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file.

        Args:
            filepath: Path to JSONL file.

        Returns:
            List of parsed JSON objects.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Training file not found: {filepath}")

        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")

        return examples

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate a training example.

        Expected format:
        {
            "instruction": "...",
            "output": "...",
            "input": "..." (optional)
        }

        Args:
            example: Training example to validate.

        Returns:
            True if valid, False otherwise.
        """
        if "instruction" not in example or "output" not in example:
            return False

        instruction = str(example.get("instruction", ""))
        output = str(example.get("output", ""))
        combined_length = len(instruction) + len(output)

        if combined_length < self.min_length or combined_length > self.max_length:
            return False

        return True

    def prepare_training_data(
        self,
        train_file: str,
        tokenizer: Any,
        deduplicate: bool = True,
    ) -> List[Any]:
        """
        Load and convert training data into Tinker Datum objects.

        Args:
            train_file: Path to training JSONL file.
            tokenizer: Tokenizer from Tinker training client.
            deduplicate: Whether to deduplicate examples.

        Returns:
            List of tinker.types.Datum objects.
        """
        if types is None:
            raise ImportError("tinker package required for data preparation")

        raw_examples = self.load_jsonl(train_file)
        print(f"Loaded {len(raw_examples)} examples from {train_file}")

        valid_examples = [ex for ex in raw_examples if self.validate_example(ex)]
        print(f"Filtered to {len(valid_examples)} valid examples")

        if deduplicate:
            seen = set()
            unique_examples = []
            for ex in valid_examples:
                key = (ex.get("instruction"), ex.get("output"))
                if key not in seen:
                    seen.add(key)
                    unique_examples.append(ex)
            print(f"Deduplicated to {len(unique_examples)} unique examples")
            valid_examples = unique_examples

        datums = []
        for ex in valid_examples:
            instruction = ex["instruction"]
            input_text = ex.get("input", "")
            output_text = ex["output"]

            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse:"
            else:
                prompt = f"{instruction}\n\nResponse:"

            full_text = f"{prompt} {output_text}"
            
            tokens = tokenizer.encode(full_text)
            if len(tokens) > self.max_seq_length:
                print(f"Warning: Skipping example with {len(tokens)} tokens (max: {self.max_seq_length})")
                continue

            datum = types.Datum(
                model_input=tokens,
                loss_fn_inputs={"target": tokens},
            )
            datums.append(datum)

        print(f"Prepared {len(datums)} training datums")
        return datums
