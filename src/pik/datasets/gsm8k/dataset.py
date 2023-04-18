from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


class GSM8KDataset(Dataset):
    """Creates a PyTorch Dataset for the GSM8K dataset.

    See: https://huggingface.co/datasets/gsm8k
    """

    def __init__(self):
        gsm8k = load_dataset("gsm8k", name="main")
        gsm8k_test = gsm8k["test"].train_test_split(10)  # type: ignore

        self.prompting = gsm8k_test["test"]  # Set aside 10 examples for prompting
        self.dataset = concatenate_datasets([gsm8k["train"], gsm8k_test["train"]])  # type: ignore

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[tuple[str, list[str]], tuple[str, list[str]]]:
        """
        Returns a tuple containing:
            question (str | list[str])
            answer (str | list[str])
        """
        datasubset = self.dataset[key]

        if isinstance(key, int):
            question = datasubset["question"]
            answer = datasubset["answer"].rsplit("#### ")[-1]
        else:
            question, answer = [], []
            for q, a in zip(datasubset["question"], datasubset["answer"]):
                question.append(q)
                answer.append(a.rsplit("#### ")[-1])

        return (question, answer)  # type: ignore
