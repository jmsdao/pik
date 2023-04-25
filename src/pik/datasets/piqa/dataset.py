from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


CHOICE_LABELS = ["(A)", "(B)"]
TEMPLATE = """{}

Choices:
{}
{}"""


class PIQADataset(Dataset):
    """Creates a PyTorch Dataset for the PIQA dataset.

    See: https://huggingface.co/datasets/piqa/
    """

    def __init__(self, choice_labels=CHOICE_LABELS, template=TEMPLATE):
        if len(choice_labels) != 2:
            raise ValueError("choice_labels must be a list of length 2")

        self.choice_labels = choice_labels
        self.template = template

        dataset = load_dataset("piqa")
        data_split = concatenate_datasets([
            dataset["train"], dataset["validation"]  # type: ignore
        ]).train_test_split(10, shuffle=False)  # Set aside 10 for prompting

        self.dataset = data_split["train"]
        self.prompting = data_split["test"]

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
            choices = [
                f"{self.choice_labels[0]} {datasubset['sol1']}",
                f"{self.choice_labels[1]} {datasubset['sol2']}",
            ]
            question = self.template.format(datasubset["goal"], *choices)
            answer = self.choice_labels[datasubset["label"]]
        else:
            zipped = zip(
                datasubset["goal"],
                datasubset["sol1"],
                datasubset["sol2"],
                datasubset["label"]
            )
            question, answer = [], []
            for g, s1, s2, lab in zipped:
                choices = [
                    f"{self.choice_labels[0]} {s1}",
                    f"{self.choice_labels[1]} {s2}",
                ]
                q = self.template.format(g, *choices)
                a = self.choice_labels[lab]
                question.append(q)
                answer.append(a)

        return (question, answer)  # type: ignore
