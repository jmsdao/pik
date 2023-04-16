from typing import Union, Iterable
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset


# Custom loading script to skip extracting train and test sets
LOADING_SCRIPT = Path(__file__).parent / "trivia_qa.py"


class TriviaQADataset(Dataset):
    """Creates a PyTorch Dataset for the TriviaQA dataset."""

    def __init__(
        self,
        loading_script=str(LOADING_SCRIPT),
        name="rc.nocontext",
        split="validation",
    ):
        if not Path(loading_script).exists():
            raise FileNotFoundError(f'Loading script not found at "{loading_script}"')

        self.dataset = load_dataset(loading_script, name=name, split=split)

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows  # type: ignore

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[
        tuple[str, list[str]],
        tuple[list[str], list[list[str]]]
    ]:
        """
        Returns a tuple containing:
            question (str | list[str]):
                text string of the question
            answer_aliases (list[str] | list[list[str]]):
                list of strings of possible answers
        """
        datasubset = self.dataset[key]  # type: ignore
        question = datasubset['question']
        if isinstance(key, int):
            answer_aliases = datasubset['answer']['normalized_aliases']  # type: ignore
        else:
            answer_aliases = [ans['normalized_aliases'] for ans in datasubset['answer']]

        return (question, answer_aliases)
