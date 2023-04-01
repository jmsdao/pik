"""
This script uses conditional imports to minimize slow imports.
"""

IMPLEMENTED_DATASETS = ["trivia_qa"]


def load_dataset_and_eval_fn(dataset_name: str) -> tuple:
    """Syntactic sugar for loading a dataset and evaluation function."""
    if dataset_name == "trivia_qa":
        from .trivia_qa import TriviaQADataset, evaluate_answer
        return TriviaQADataset(), evaluate_answer

    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not implemented."
        f"Implemented datasets: {IMPLEMENTED_DATASETS}"
    )
