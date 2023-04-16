# This script uses conditional imports to minimize slow imports.

IMPLEMENTED_DATASETS = ["lambada", "trivia_qa"]


def load_dataset(dataset_name: str):
    """One-liner for loading a dataset."""
    if dataset_name == "lambada":
        from .lambada import LambadaDataset
        return LambadaDataset()

    if dataset_name == "trivia_qa":
        from .trivia_qa import TriviaQADataset
        return TriviaQADataset()

    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not implemented."
        f"Implemented datasets: {IMPLEMENTED_DATASETS}"
    )


def get_eval_fn(dataset_name: str):
    """One-liner for instantiating an evaluation function for a given dataset."""
    if dataset_name == "lambada":
        from .lambada import evaluate_answer
        return evaluate_answer

    if dataset_name == "trivia_qa":
        from .trivia_qa import evaluate_answer
        return evaluate_answer

    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not implemented."
        f"Implemented datasets: {IMPLEMENTED_DATASETS}"
    )
