import re
import string
from typing import Union, List


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


def evaluate_answer(model_answer: str, dataset_aliases: Union[str, List[str]]) -> int:
    """Evaluate if model answer is correct against possible answer aliases.

    Args:
        model_answer (str): model's answer
        dataset_aliases (str or list[str]): possible answer aliases from dataset

    Returns:
        result (int): 1 if model answer is correct, 0 otherwise
    """
    result = 0
    model_answer_normed = normalize_answer(model_answer)

    if isinstance(dataset_aliases, str):
        dataset_aliases = [dataset_aliases]

    for alias in dataset_aliases:
        if alias in model_answer_normed:
            result = 1
            break

    return result
