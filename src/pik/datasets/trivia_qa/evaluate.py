import re
import string
from typing import Union


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Taken from official triviaqa eval script:
    https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def evaluate_answer(
    model_answers: Union[str, list[str]],
    dataset_answers: Union[str, list[str]]
) -> list[int]:
    """Evaluate if model answer(s) is correct. An answer is correct if any of the
    possible answer aliases is a substring of the model answer.

    Args:
        model_answers (str or list[str]): model answer(s)
        dataset_answers (str or list[str]): possible answer aliases from dataset

    Returns:
        results (list[int]): 1 if model answer is correct, 0 otherwise
    """
    if isinstance(model_answers, str):
        model_answers = [model_answers]
    if isinstance(dataset_answers, str):
        dataset_answers = [dataset_answers]

    model_answers_normed = [normalize_answer(a) for a in model_answers]

    results = []
    for answer in model_answers_normed:
        for alias in dataset_answers:
            if alias in answer:
                results.append(1)
                break
        else:
            results.append(0)

    return results
