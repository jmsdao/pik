# This script uses conditional imports to minimize slow imports.

IMPLEMENTED_DATASETS = [
    "gsm8k", "mbpp", "lambada", "random_arithemtic", "trivia_qa"
]


def load_dataset(dataset_name: str):
    """One-liner for loading a dataset."""
    if dataset_name == "gsm8k":
        from .gsm8k import GSM8KDataset
        return GSM8KDataset()

    if dataset_name == "lambada":
        from .lambada import LAMBADADataset
        return LAMBADADataset()

    if dataset_name == "mbpp":
        from .mbpp import MBPPDataset
        return MBPPDataset()

    if dataset_name == "random_arithmetic":
        from .random_arithmetic import RandomArithmeticDataset
        return RandomArithmeticDataset()

    if dataset_name == "trivia_qa":
        from .trivia_qa import TriviaQADataset
        return TriviaQADataset()

    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not implemented. "
        f"Implemented datasets: {IMPLEMENTED_DATASETS}"
    )


def get_eval_fn(dataset_name: str):
    """One-liner for instantiating an evaluation function for a given dataset."""
    if dataset_name == "gsm8k":
        from .gsm8k import evaluate_answer
        return evaluate_answer

    if dataset_name == "lambada":
        from .lambada import evaluate_answer
        return evaluate_answer

    if dataset_name == "mbpp":
        from .mbpp import evaluate_answer
        return evaluate_answer

    if dataset_name == "random_arithmetic":
        from .random_arithmetic import evaluate_answer
        return evaluate_answer

    if dataset_name == "trivia_qa":
        from .trivia_qa import evaluate_answer
        return evaluate_answer

    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not implemented. "
        f"Implemented datasets: {IMPLEMENTED_DATASETS}"
    )
