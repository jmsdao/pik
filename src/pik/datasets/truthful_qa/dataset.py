import random
from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset


TEMPLATE = """{}

Choices:
{}"""


class TruthfulQADataset(Dataset):
    """Creates a PyTorch Dataset for the TruthfulQA dataset.

    See: https://huggingface.co/datasets/truthful_qa/
    """

    def __init__(self, template=TEMPLATE, choice_shuffle_seed=42):
        self.template = template
        self.choice_shuffle_seed = choice_shuffle_seed

        dataset = load_dataset("truthful_qa", name="multiple_choice")
        # Set aside 10 for prompting
        data_split = dataset["validation"].train_test_split(10, shuffle=False)  # type: ignore

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
            q = datasubset["question"]
            chs = datasubset["mc2_targets"]["choices"]
            labs = datasubset["mc2_targets"]["labels"]

            # Shuffle options since default is having answers at the start of choices
            random.seed(self.choice_shuffle_seed)
            random.shuffle(chs)
            random.seed(self.choice_shuffle_seed)
            random.shuffle(labs)

            choices = "\n".join([f"({i+1}) {ch}" for i, ch in enumerate(chs)])
            question = self.template.format(q, choices)
            answer = ";".join([f"({i+1})" for i, lab in enumerate(labs) if lab == 1])
        else:
            zipped = zip(
                datasubset["question"],
                datasubset["mc2_targets"]["choices"],
                datasubset["mc2_targets"]["labels"],
            )
            question, answer = [], []
            for q, chs, labs in zipped:
                random.seed(self.choice_shuffle_seed)
                random.shuffle(chs)
                random.seed(self.choice_shuffle_seed)
                random.shuffle(labs)

                choices = "\n".join([f"({i+1}) {ch}" for i, ch in enumerate(chs)])
                q = self.template.format(q, choices)
                a = ";".join([f"({i+1})" for i, lab in enumerate(labs) if lab == 1])
                question.append(q)
                answer.append(a)

        return (question, answer)  # type: ignore
