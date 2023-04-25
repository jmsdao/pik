# This script uses conditional imports to minimize slow imports.

IMPLEMENTED_DATASETS = [
    "gsm8k", "lambada", "mbpp", "mmlu", "openbookqa", "piqa",
    "random_arithemtic", "social_i_qa", "trivia_qa", "truthful_qa",
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

    if dataset_name == "mmlu":
        from .mmlu import MMLUDataset
        return MMLUDataset()

    if dataset_name == "openbookqa":
        from .openbookqa import OpenBookQADataset
        return OpenBookQADataset()

    if dataset_name == "piqa":
        from .piqa import PIQADataset
        return PIQADataset()

    if dataset_name == "random_arithmetic":
        from .random_arithmetic import RandomArithmeticDataset
        return RandomArithmeticDataset()

    if dataset_name == "social_i_qa":
        from .social_i_qa import SIQADataset
        return SIQADataset()

    if dataset_name == "trivia_qa":
        from .trivia_qa import TriviaQADataset
        return TriviaQADataset()

    if dataset_name == "truthful_qa":
        from .truthful_qa import TruthfulQADataset
        return TruthfulQADataset()

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

    if dataset_name == "mmlu":
        from .mmlu import evaluate_answer
        return evaluate_answer

    if dataset_name == "openbookqa":
        from .openbookqa import evaluate_answer
        return evaluate_answer

    if dataset_name == "piqa":
        from .piqa import evaluate_answer
        return evaluate_answer

    if dataset_name == "random_arithmetic":
        from .random_arithmetic import evaluate_answer
        return evaluate_answer

    if dataset_name == "social_i_qa":
        from .social_i_qa import evaluate_answer
        return evaluate_answer

    if dataset_name == "trivia_qa":
        from .trivia_qa import evaluate_answer
        return evaluate_answer

    if dataset_name == "truthful_qa":
        from .truthful_qa import evaluate_answer
        return evaluate_answer

    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not implemented. "
        f"Implemented datasets: {IMPLEMENTED_DATASETS}"
    )
