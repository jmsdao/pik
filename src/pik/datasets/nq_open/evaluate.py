import re
import string
from typing import Union


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Taken from official triviaqa eval script:
    https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def evaluate_answer(
    model_answers: Union[str, list[str]],
    dataset_answers: Union[list[str], list[list[str]]],
) -> list[int]:
    """Evaluate if model answer is correct. An answer is correct if any of the
    possible answer aliases is a substring of the model answer.

    Args:
        model_answers (str or list[str]): model answer(s)
        dataset_answers (list[str] or list[list[str]]): possible answer aliases from dataset

    Returns:
        results (list[int]): 1 if model answer is correct, 0 otherwise
    """
    if isinstance(model_answers, str) and not isinstance(dataset_answers[0], str):
        raise ValueError("If model_answers is str, dataset_answers expects list[str].")
    if (isinstance(model_answers, list) and isinstance(model_answers[0], str)) and not (
        isinstance(dataset_answers[0], list) and isinstance(dataset_answers[0][0], str)
    ):
        raise ValueError(
            "If model_answers is list[str], dataset_answers expects list[list[str]]."
        )
    if isinstance(model_answers, list) and len(model_answers) != len(dataset_answers):
        raise ValueError(
            "If model_answers is list[str], then len(model_answers) should equal len(dataset_answers)."
        )

    if isinstance(model_answers, str):
        model_answers = [model_answers]
        dataset_answers = [dataset_answers]  # type: ignore

    results = []
    for model_answer, dataset_answer in zip(model_answers, dataset_answers):
        model_answer = normalize_answer(model_answer)
        aliases = dataset_answer.split("###")  # type: ignore
        if any(normalize_answer(alias) in model_answer for alias in aliases):
            results.append(1)
        else:
            results.append(0)

    return results
