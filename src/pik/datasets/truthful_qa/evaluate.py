from typing import Union


def evaluate_answer(
    model_answers: Union[str, list[str]],
    dataset_answers: Union[str, list[str]],
) -> list[int]:
    """Evaluate if model answer is correct. An answer is correct if any of
    the possible dataset answers is a substring of the model answer.

    Args:
        model_answers (str or list[str]): model answer(s)
        dataset_answers (str or list[str]): answer(s) from the dataset

    Returns:
        results (list[int]): 1 if model answer is correct, 0 otherwise
    """
    if isinstance(model_answers, str) and not isinstance(dataset_answers, str):
        raise ValueError("If model_answers is str, dataset_answers expects str.")
    if (isinstance(model_answers, list) and isinstance(model_answers[0], str)) and not (
        isinstance(dataset_answers, list) and isinstance(dataset_answers[0], str)
    ):
        raise ValueError(
            "If model_answers is list[str], dataset_answers expects list[str]."
        )
    if isinstance(model_answers, list) and len(model_answers) != len(dataset_answers):
        raise ValueError(
            "If model_answers is list[str], then len(model_answers) should equal len(dataset_answers)."
        )

    if isinstance(model_answers, str) and isinstance(dataset_answers, str):
        model_answers = [model_answers]
        dataset_answers = [dataset_answers]

    results = []
    for model_answer, dataset_answer in zip(model_answers, dataset_answers):
        if any(a in model_answer for a in dataset_answer.split(";")):
            results.append(1)
        else:
            results.append(0)

    return results
