from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


class LambadaDataset(Dataset):
    """Creates a PyTorch Dataset for the Lambada dataset."""

    def __init__(self):
        lambada = load_dataset("lambada")
        self.dataset = concatenate_datasets([lambada["validation"], lambada["test"]])  # type: ignore

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[tuple[str, list[str]], tuple[str, list[str]]]:
        """
        Returns a tuple:
            passage (str | list[str]):
                a passage with the last word omitted
            target (str | list[str]):
                the target word that completes the passage
        """
        datasubset = self.dataset[key]["text"]

        if isinstance(key, int):
            passage, target = datasubset.rsplit(" ", 1)
        else:
            passage, target = [], []
            for data in datasubset:
                p, t = data.rsplit(" ", 1)
                passage.append(p)
                target.append(t)

        return (passage, target)  # type: ignore
