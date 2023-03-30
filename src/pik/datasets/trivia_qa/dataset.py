from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset


# Custom loading script to skip extracting train and test sets
LOADING_SCRIPT = Path(__file__).parent / "loading.py"


class TriviaQADataset(Dataset):
    """Creates a PyTorch Dataset for the TriviaQA dataset."""
    def __init__(self,
        loading_script=str(LOADING_SCRIPT),
        name='rc.nocontext',
        split='validation',
    ):
        if not Path(loading_script).exists():
            raise FileNotFoundError(f'Loading script not found at "{loading_script}"')

        self.dataset = load_dataset(
            loading_script,
            name=name,
            split=split,
        )

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows  # type: ignore

    def __getitem__(self, key: int):
        """
        Returns a tuple: (question, answer_aliases)
            question (str): text string of the question
            answer_aliases (list[str]): list of strings of possible answers
        """
        return (
            self.dataset[key]['question'],  # type: ignore
            self.dataset[key]['answer']['normalized_aliases']  # type: ignore
        )
