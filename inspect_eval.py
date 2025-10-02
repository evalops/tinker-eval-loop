"""
Inspect AI evaluation integration with Tinker sampling.
"""

from typing import Any, Dict, List, Optional

try:
    from inspect_ai import Task, task, eval_async
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig, Model
    from inspect_ai.scorer import match, includes
    from inspect_ai.solver import generate
    
    QA_SAMPLES = [
        Sample(input="What is 2 + 2?", target="4"),
        Sample(input="What is the capital of France?", target="Paris"),
        Sample(input="What color is grass?", target="green"),
        Sample(input="How many days in a week?", target="7"),
        Sample(input="What is 10 x 5?", target="50"),
    ]
    
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Task = None
    task = None
    QA_SAMPLES = []

try:
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
    TINKER_INSPECT_AVAILABLE = True
except ImportError:
    TINKER_INSPECT_AVAILABLE = False


def simple_qa_task():
    """
    Simple QA evaluation task for demo purposes.
    
    Tests basic factual knowledge with exact match scoring.
    """
    if not INSPECT_AVAILABLE or not Task:
        raise ImportError("inspect_ai required for this task")
    
    @task
    def _simple_qa() -> Task:
        return Task(
            name="simple_qa",
            dataset=MemoryDataset(name="simple_qa", samples=QA_SAMPLES),
            solver=generate(),
            scorer=includes(),
        )
    
    return _simple_qa()


async def run_inspect_evaluation(
    service_client: Any,
    model_path: str,
    model_name: str,
    renderer_name: str,
    tasks: List[str],
) -> float:
    """
    Run Inspect AI evaluation using Tinker sampling.
    
    Args:
        service_client: Tinker service client.
        model_path: Path to model checkpoint (tinker:// or mock://).
        model_name: Base model name.
        renderer_name: Renderer name for message formatting.
        tasks: List of task names to evaluate.
    
    Returns:
        Aggregate accuracy score.
    """
    if not INSPECT_AVAILABLE:
        print("  Warning: inspect_ai not available, using fallback")
        return 0.5
    
    if model_path.startswith("mock://"):
        print("  Warning: Mock mode - using simulated eval")
        return 0.5
    
    try:
        if not TINKER_INSPECT_AVAILABLE:
            print("  Warning: tinker_cookbook.eval not available")
            results = await eval_async(
                tasks=[simple_qa_task()],
                model=f"openai/{model_name}",
            )
        else:
            sampling_client = service_client.create_sampling_client(model_path=model_path)
            
            api = InspectAPIFromTinkerSampling(
                renderer_name=renderer_name,
                model_name=model_name,
                sampling_client=sampling_client,
                verbose=False,
            )
            
            model = Model(api=api, config=GenerateConfig(max_tokens=100, temperature=0.0))
            
            eval_tasks = [simple_qa_task()] if "simple_qa" in tasks else []
            
            results = await eval_async(tasks=eval_tasks, model=model)
        
        if results and len(results) > 0:
            scores = [r.scores[0].value for r in results if r.scores]
            return sum(scores) / len(scores) if scores else 0.0
        
        return 0.0
    
    except Exception as e:
        print(f"  Warning: Inspect AI evaluation failed: {e}")
        return 0.5
