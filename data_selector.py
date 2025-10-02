"""
Data selection utilities for mining hard examples based on evaluation failures.
"""

from typing import Any, Dict, List, Optional


class DataSelector:
    """Select additional training examples based on evaluation failures."""

    def __init__(self, corpus_path: Optional[str] = None):
        """
        Initialize data selector.

        Args:
            corpus_path: Optional path to additional data corpus for mining.
        """
        self.corpus_path = corpus_path

    def analyze_failures(
        self, eval_results: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Analyze evaluation results to identify failure patterns.

        Args:
            eval_results: Dictionary containing evaluation results with per-task breakdowns.

        Returns:
            Dictionary mapping failure categories to lists of example IDs or topics.
        """
        failure_patterns = {
            "low_accuracy_tasks": [],
            "high_error_rate_tasks": [],
            "specific_topics": [],
        }

        tasks = eval_results.get("tasks", {})
        for task_name, task_results in tasks.items():
            accuracy = task_results.get("accuracy", 1.0)
            error_rate = task_results.get("error_rate", 0.0)

            if accuracy < 0.7:
                failure_patterns["low_accuracy_tasks"].append(task_name)

            if error_rate > 0.2:
                failure_patterns["high_error_rate_tasks"].append(task_name)

            failed_topics = task_results.get("failed_topics", [])
            failure_patterns["specific_topics"].extend(failed_topics)

        return failure_patterns

    def select_additional_examples(
        self,
        failure_patterns: Dict[str, List[str]],
        num_examples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Select additional training examples based on failure patterns.

        Args:
            failure_patterns: Dictionary of failure categories from analyze_failures.
            num_examples: Maximum number of additional examples to select.

        Returns:
            List of selected training examples in instruction/output format.
        """
        if not self.corpus_path:
            print(
                "Warning: No corpus path configured. Cannot mine additional examples."
            )
            return []

        selected_examples = []

        for category, items in failure_patterns.items():
            if not items:
                continue

            print(f"Mining examples for {category}: {items}")

        print(
            f"Selected {len(selected_examples)} additional examples (placeholder implementation)"
        )
        return selected_examples

    def reweight_dataset(
        self,
        current_examples: List[Dict[str, Any]],
        failure_patterns: Dict[str, List[str]],
        boost_factor: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Reweight existing examples to emphasize failure categories.

        Args:
            current_examples: Current training dataset.
            failure_patterns: Failure categories to boost.
            boost_factor: Multiplier for examples in failure categories.

        Returns:
            Reweighted dataset (may include duplicates for emphasis).
        """
        reweighted = list(current_examples)

        for example in current_examples:
            category = example.get("category") or example.get("topic")
            if category in failure_patterns.get("specific_topics", []):
                for _ in range(int(boost_factor) - 1):
                    reweighted.append(example)

        print(
            f"Reweighted dataset from {len(current_examples)} to {len(reweighted)} examples"
        )
        return reweighted
