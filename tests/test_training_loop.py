"""
Integration tests for the training loop.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path

import pytest

from trainer_with_eval import async_main


@pytest.mark.asyncio
class TestTrainingLoop:
    """Test suite for training loop integration."""

    async def test_early_stopping_on_threshold_met(self, tmp_path):
        """Training stops early when eval threshold is met in first round."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config_file = tmp_path / "config.json"
        config_file.write_text(
            f'{{'
            f'"base_model": "test-model", '
            f'"train_file": "{train_file}", '
            f'"max_rounds": 5, '
            f'"eval_threshold": 0.8'
            f'}}'
        )

        mock_client = MagicMock()
        mock_training_client = MagicMock()
        mock_client.create_lora_training_client.return_value = mock_training_client
        mock_training_client.get_tokenizer.return_value = MagicMock()
        mock_training_client.save_state.return_value = "tinker://checkpoint-1"

        with patch("trainer_with_eval.tinker.ServiceClient", return_value=mock_client):
            with patch("trainer_with_eval.prepare_training_data", return_value=[MagicMock()]):
                with patch("trainer_with_eval.run_evaluations", new=AsyncMock(return_value=0.85)):
                    await async_main(str(config_file))

        assert mock_training_client.save_state.call_count == 1

    async def test_full_rounds_below_threshold(self, tmp_path):
        """Training runs all rounds when threshold never met."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config_file = tmp_path / "config.json"
        config_file.write_text(
            f'{{'
            f'"base_model": "test-model", '
            f'"train_file": "{train_file}", '
            f'"max_rounds": 3, '
            f'"eval_threshold": 0.9, '
            f'"lr_decay": 0.5'
            f'}}'
        )

        mock_client = MagicMock()
        mock_training_client = MagicMock()
        mock_client.create_lora_training_client.return_value = mock_training_client
        mock_training_client.get_tokenizer.return_value = MagicMock()
        mock_training_client.save_state.return_value = "tinker://checkpoint"

        with patch("trainer_with_eval.tinker.ServiceClient", return_value=mock_client):
            with patch("trainer_with_eval.prepare_training_data", return_value=[MagicMock()]):
                with patch("trainer_with_eval.run_evaluations", new=AsyncMock(return_value=0.7)):
                    await async_main(str(config_file))

        assert mock_training_client.save_state.call_count == 3

    async def test_evalops_integration_called(self, tmp_path):
        """EvalOps client is called when enabled."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config_file = tmp_path / "config.json"
        config_file.write_text(
            f'{{'
            f'"base_model": "test-model", '
            f'"train_file": "{train_file}", '
            f'"max_rounds": 1, '
            f'"evalops_enabled": true, '
            f'"evalops_test_suite_id": "suite-123"'
            f'}}'
        )

        mock_evalops_client = AsyncMock()
        mock_evalops_client.submit_training_results.return_value = {"data": {"id": "run-123"}}
        mock_evalops_client.close = AsyncMock()

        mock_tinker_client = MagicMock()
        mock_training_client = MagicMock()
        mock_tinker_client.create_lora_training_client.return_value = mock_training_client
        mock_training_client.get_tokenizer.return_value = MagicMock()
        mock_training_client.save_state.return_value = "tinker://checkpoint"

        with patch("trainer_with_eval.tinker.ServiceClient", return_value=mock_tinker_client):
            with patch("trainer_with_eval.prepare_training_data", return_value=[MagicMock()]):
                with patch("trainer_with_eval.run_evaluations", new=AsyncMock(return_value=0.9)):
                    with patch("trainer_with_eval.EvalOpsClient", return_value=mock_evalops_client):
                        await async_main(str(config_file))

        mock_evalops_client.submit_training_results.assert_called_once()
        mock_evalops_client.close.assert_called_once()

    async def test_lr_decay_across_rounds(self, tmp_path):
        """Learning rate decays correctly across rounds."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"instruction": "test", "output": "result"}\n')

        config_file = tmp_path / "config.json"
        config_file.write_text(
            f'{{'
            f'"base_model": "test-model", '
            f'"train_file": "{train_file}", '
            f'"max_rounds": 3, '
            f'"learning_rate": 1.0, '
            f'"lr_decay": 0.5, '
            f'"eval_threshold": 0.99'
            f'}}'
        )

        observed_lrs = []

        def mock_training_round(client, datums, lr):
            observed_lrs.append(lr)

        mock_client = MagicMock()
        mock_training_client = MagicMock()
        mock_client.create_lora_training_client.return_value = mock_training_client
        mock_training_client.get_tokenizer.return_value = MagicMock()
        mock_training_client.save_state.return_value = "tinker://checkpoint"

        with patch("trainer_with_eval.tinker.ServiceClient", return_value=mock_client):
            with patch("trainer_with_eval.prepare_training_data", return_value=[MagicMock()]):
                with patch("trainer_with_eval.run_evaluations", new=AsyncMock(return_value=0.7)):
                    with patch("trainer_with_eval.run_training_round", side_effect=mock_training_round):
                        await async_main(str(config_file))

        assert len(observed_lrs) == 3
        assert observed_lrs[0] == 1.0
        assert observed_lrs[1] == 0.5
        assert observed_lrs[2] == 0.25
