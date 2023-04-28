from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


class NQOpenDataset(Dataset):
    """Creates a PyTorch Dataset for the NQOpen dataset.

    See: https://huggingface.co/datasets/nq_open/
    """

    def __init__(self):
        dataset = load_dataset("nq_open")
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
            answer (str | list[str]): note that multiple possible answers are
                delimited by "###".
        """
        datasubset = self.dataset[key]

        if isinstance(key, int):
            question = datasubset["question"]
            answer = "###".join(datasubset["answer"])
        else:
            question, answer = [], []
            for q, a in zip(datasubset["question"], datasubset["answer"]):
                question.append(q)
                answer.append("###".join(a))

        return (question, answer)  # type: ignore
