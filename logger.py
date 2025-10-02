"""
Structured logging utilities for the training loop.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """JSON-structured logger for training metrics and events."""
    
    def __init__(self, run_dir: Optional[Path] = None, log_level: int = logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            run_dir: Directory to write metrics.jsonl. If None, logs to stdout only.
            log_level: Logging level (default INFO).
        """
        self.run_dir = run_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = run_dir / "metrics.jsonl"
        else:
            self.metrics_file = None
        
        self.logger = logging.getLogger("tinker_eval_loop")
        self.logger.setLevel(log_level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: str = "INFO"):
        """
        Log a structured event.
        
        Args:
            event_type: Type of event (e.g., "training_start", "eval_complete").
            data: Event data dictionary.
            level: Log level (INFO, WARNING, ERROR).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            **data
        }
        
        if self.metrics_file:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(f"[{event_type}] {json.dumps(data, indent=None)}")
    
    def log_training_step(self, round_num: int, step: int, lr: float, loss: Optional[float] = None):
        """Log a training step."""
        self.log_event("training_step", {
            "round": round_num,
            "step": step,
            "learning_rate": lr,
            "loss": loss,
        })
    
    def log_evaluation(self, round_num: int, score: float, threshold: float, passed: bool, metrics: Optional[Dict] = None):
        """Log an evaluation result."""
        self.log_event("evaluation", {
            "round": round_num,
            "score": score,
            "threshold": threshold,
            "passed": passed,
            "metrics": metrics or {},
        })
    
    def log_checkpoint(self, round_num: int, checkpoint_uri: str):
        """Log a checkpoint save."""
        self.log_event("checkpoint", {
            "round": round_num,
            "checkpoint_uri": str(checkpoint_uri),
        })
    
    def log_config(self, config: Dict[str, Any]):
        """Log the run configuration."""
        self.log_event("config", config)
