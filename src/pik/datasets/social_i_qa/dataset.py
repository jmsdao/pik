from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


CHOICE_LABELS = ["(A)", "(B)", "(C)"]
TEMPLATE = """{}

{}

Choices:
{}
{}
{}"""


class SIQADataset(Dataset):
    """Creates a PyTorch Dataset for the SIQA dataset.

    See: https://huggingface.co/datasets/social_i_qa/
    """
    def __init__(self, choice_labels=CHOICE_LABELS, template=TEMPLATE):
        if len(choice_labels) != 3:
            raise ValueError("choice_labels must be a list of length 3")

        self.choice_labels = choice_labels
        self.template = template

        dataset = load_dataset("social_i_qa")
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
                f"{self.choice_labels[0]} {datasubset['answerA']}",
                f"{self.choice_labels[1]} {datasubset['answerB']}",
                f"{self.choice_labels[2]} {datasubset['answerC']}",
            ]
            question = self.template.format(
                datasubset["context"], datasubset["question"], *choices
            )
            answer = self.choice_labels[int(datasubset["label"])]
        else:
            zipped = zip(
                datasubset["context"],
                datasubset["question"],
                datasubset["answerA"],
                datasubset["answerB"],
                datasubset["answerC"],
                datasubset["label"],
            )
            question, answer = [], []
            for c, q, ans_a, ans_b, ans_c, lab in zipped:
                choices = [
                    f"{self.choice_labels[0]} {ans_a}",
                    f"{self.choice_labels[1]} {ans_b}",
                    f"{self.choice_labels[2]} {ans_c}",
                ]
                q = self.template.format(c, q, *choices)
                a = self.choice_labels[int(lab)]
                question.append(q)
                answer.append(a)

        return (question, answer)  # type: ignore
