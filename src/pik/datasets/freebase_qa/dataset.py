from typing import Union, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


class FreebaseQADataset(Dataset):
    """Creates a PyTorch Dataset for the FreebaseQA dataset.

    See: https://huggingface.co/datasets/freebase_qa/
    """

    def __init__(self):
        dataset = load_dataset("freebase_qa")
        data_split = concatenate_datasets([
            dataset["train"], dataset["validation"], dataset["test"]  # type: ignore
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
            question = datasubset["RawQuestion"]
            answer = datasubset["Parses"]["Answers"][0]["AnswersName"][0][0]
        else:
            zipped = zip(
                datasubset["RawQuestion"],
                datasubset["Parses"],
            )
            question, answer = [], []
            for q, a in zipped:
                question.append(q)
                answer.append(a["Answers"][0]["AnswersName"][0][0])

        return (question, answer)  # type: ignore
