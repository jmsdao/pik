from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


KEY2INT = {"A": 0, "B": 1, "C": 2, "D": 3}
CHOICE_LABELS = ["(A)", "(B)", "(C)", "(D)"]
TEMPLATE = """Fact:
{}

Question:
{}

Choices:
{}
{}
{}
{}"""


class OpenBookQADataset(Dataset):
    """Creates a PyTorch Dataset for the OpenBookQA dataset.

    See: https://huggingface.co/datasets/openbookqa/
    """
    def __init__(self, choice_labels=CHOICE_LABELS, template=TEMPLATE):
        if len(choice_labels) != 4:
            raise ValueError("choice_labels must have 4 elements")

        self.choice_labels = choice_labels
        self.template = template

        datasets = load_dataset("openbookqa", name="additional")
        datasets = concatenate_datasets([
            datasets["train"], datasets["validation"], datasets["test"]  # type: ignore
        ]).train_test_split(10, shuffle=False)  # Set aside 10 for prompting
        self.dataset = datasets["train"]
        self.prompting = datasets["test"]

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[tuple[str, str], tuple[list[str], list[str]]]:
        """
        Returns a tuple containing:
            question (str | list[str])
            answer (str | list[str])
        """
        datasubset = self.dataset[key]

        if isinstance(key, int):
            zipped = zip(self.choice_labels, datasubset["choices"]["text"])
            choices = [f"{lab} {ch}" for lab, ch in zipped]
            question = self.template.format(
                datasubset["fact1"], datasubset["question_stem"], *choices
            )
            answer = self.choice_labels[KEY2INT[datasubset["answerKey"]]]
        else:
            zipped = zip(
                datasubset["fact1"],
                datasubset["question_stem"],
                datasubset["choices"],
                datasubset["answerKey"]
            )
            question, answer = [], []
            for f, q, c, a in zipped:
                choices = [
                    f"{lab} {ch}" for lab, ch in zip(self.choice_labels, c["text"])
                ]
                q = self.template.format(f, q, *choices)
                a = self.choice_labels[KEY2INT[a]]
                question.append(q)
                answer.append(a)

        return (question, answer)  # type: ignore
