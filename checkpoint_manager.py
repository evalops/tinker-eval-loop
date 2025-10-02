"""
Checkpoint and resume management for training runs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class CheckpointManager:
    """Manage training checkpoints and resume state."""
    
    def __init__(self, run_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            run_dir: Directory to store checkpoints and state.
        """
        self.run_dir = run_dir
        self.state_file = run_dir / "run_state.json"
    
    def save_run_state(
        self,
        round_idx: int,
        global_step: int,
        learning_rate: float,
        checkpoint_uri: str,
        config: Dict[str, Any],
    ) -> None:
        """
        Save current run state for resumption.
        
        Args:
            round_idx: Current training round.
            global_step: Current global step count.
            learning_rate: Current learning rate.
            checkpoint_uri: URI of the latest checkpoint.
            config: Full configuration dict.
        """
        state = {
            "round_idx": round_idx,
            "global_step": global_step,
            "learning_rate": learning_rate,
            "checkpoint_uri": checkpoint_uri,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_run_state(self) -> Optional[Dict[str, Any]]:
        """
        Load saved run state.
        
        Returns:
            Saved state dict, or None if no state file exists.
        """
        if not self.state_file.exists():
            return None
        
        with open(self.state_file, "r") as f:
            return json.load(f)
    
    def has_saved_state(self) -> bool:
        """Check if a saved state exists."""
        return self.state_file.exists()


def find_latest_run() -> Optional[Path]:
    """
    Find the most recent run directory.
    
    Returns:
        Path to latest run directory, or None if no runs exist.
    """
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    return run_dirs[0] if run_dirs else None
