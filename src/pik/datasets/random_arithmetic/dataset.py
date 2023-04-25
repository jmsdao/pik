from typing import Iterable, Union

import numpy as np
from torch.utils.data import Dataset


def rand_int(n_digits: int = 1) -> int:
    """Returns a random integer with `n_digits` digits."""
    if n_digits == 1:
        num = np.random.randint(0, 10)
    else:
        num = np.random.randint(10 ** (n_digits - 1), 10**n_digits)

    return num


def int2str(i: int) -> str:
    """Converts an integer to a string with commas every 3 digits."""
    s = []
    for pos, digit in enumerate(reversed(str(abs(i)))):
        s.append(digit)
        if (pos + 1) % 3 == 0:
            s.append(",")

    s = s[::-1]
    if s[0] == ",":
        s = s[1:]
    s = "".join(s)

    if i < 0:
        s = "-" + s

    return s


def add_digits(n_digits: int, comma: bool = True, **kwargs) -> tuple[str, str]:
    """Returns a question and answer for a random addition problem.

    Eg. "What is 4 plus 9?" -> "13"
    """
    fmt = int2str if comma else lambda x: x
    i1 = rand_int(n_digits)
    i2 = rand_int(n_digits)
    question = f"What is {fmt(i1)} plus {fmt(i2)}?"
    answer = fmt(i1 + i2)

    return question, answer


def multiop_digits(n_digits: int, comma: bool = True, **kwargs) -> tuple[str, str]:
    """Returns a question and answer for a random multi-operation problem.

    Eg. " What is 5 * 1 - 6?" -> "-1"
    """
    fmt = int2str if comma else lambda x: x
    i1 = rand_int(n_digits)
    i2 = rand_int(n_digits)
    i3 = rand_int(n_digits)
    plus_or_minus = np.random.choice(["+", "-"], 1)[0]

    equation = f"{i1} * {i2} {plus_or_minus} {i3}"
    scope = {}
    exec(f"answer = {equation}", scope)

    question = f"What is {fmt(i1)} * {fmt(i2)} {plus_or_minus} {fmt(i3)}?"
    answer = fmt(scope["answer"])

    return question, answer


def multiply_digits(n_digits: int, comma: bool = True, **kwargs) -> tuple[str, str]:
    """Returns a question and answer for a random multiplication problem.

    Eg. "What is 4 times 9?" -> "36"
    """
    fmt = int2str if comma else lambda x: x
    i1 = rand_int(n_digits)
    i2 = rand_int(n_digits)
    question = f"What is {fmt(i1)} times {fmt(i2)}?"
    answer = fmt(i1 * i2)

    return question, answer


# def multisum_digits(n_digits, comma=True, n_terms=5, **kwargs):
def multisum_digits(
    n_digits: int,
    comma: bool = True,
    n_terms: int = 5,
    **kwargs,
) -> tuple[str, str]:
    """Returns a question and answer for a random multi-sum problem.

    Eg. "What is 4 + 9 + 2?" -> "15"
    """
    fmt = int2str if comma else lambda x: x
    i = [rand_int(n_digits) for _ in range(n_terms)]
    question = f"What is {fmt(i[0])}"
    for i_ in i[1:]:
        question += f" + {fmt(i_)}"
    question += "?"
    answer = fmt(sum(i))

    return question, answer


def subtract_digits(n_digits: int, comma: bool = True, **kwargs) -> tuple[str, str]:
    """Returns a question and answer for a random subtraction problem.

    Eg. "What is 4 minus 9?" -> "-5"
    """
    fmt = int2str if comma else lambda x: x
    i1 = rand_int(n_digits)
    i2 = rand_int(n_digits)
    question = f"What is {fmt(i1)} minus {fmt(i2)}?"
    answer = fmt(i1 - i2)

    return question, answer


class RandomArithmeticDataset(Dataset):
    """Dataset for random arithmetic problems.

    Inspired by section D of this paper: https://arxiv.org/abs/2207.05221
    """

    def __init__(self, n_samples=20000, comma_prob=0.7, seed_delta=0):
        """The index of the dataset corresponds to the seed used to generate
        the problem. You can set `seed_delta` to change the starting seed.
        """
        self.n_samples = n_samples
        self.comma_prob = comma_prob
        self.seed_delta = seed_delta
        self.fnc_list = [
            add_digits,
            multiop_digits,
            multiply_digits,
            multisum_digits,
            subtract_digits,
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[tuple[str, str], tuple[list[str], list[str]]]:
        """
        Returns a tuple containing:
            question (str | list[str])
            answer (str | list[str])
        """

        # Convert various input types into list of indices
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))
        if isinstance(key, Iterable):
            key = list(key)
        if isinstance(key, int):
            key = [key]

        # Check for invalid indices (positive or negative)
        if any([k < -len(self) or k >= len(self) for k in key]):
            raise IndexError("Index out of range")

        # Convert negative indices to positive
        key = [k if k >= 0 else k + len(self) for k in key]

        # Generate questions and answers
        questions, answers = [], []
        for i in key:
            np.random.seed(i + self.seed_delta)
            kwargs = {
                "comma": np.random.rand() < self.comma_prob,
                "n_terms": np.random.randint(3, 6),
            }
            fnc = np.random.choice(self.fnc_list)
            n_digits = np.random.randint(1, 6)
            question, answer = fnc(n_digits, **kwargs)
            questions.append(question)
            answers.append(answer)

        # Unlist if single index
        if len(key) == 1:
            questions = questions[0]
            answers = answers[0]

        return questions, answers  # type: ignore
