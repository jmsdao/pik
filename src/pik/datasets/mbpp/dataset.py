from typing import Iterable, Union
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset

PROMPT_TEMPLATE = """{prompt}

Tests to pass:
### START TEST CODE
{test_code}
### END TEST CODE
"""


def prep_prompt(
    prompt: str,
    test_imports: str,
    test_list: str,
    template: str = PROMPT_TEMPLATE,
) -> str:
    test_code = "\n".join(test_imports + test_list)
    return template.format(prompt=prompt, test_code=test_code)


class MBPPDataset(Dataset):
    """Creates a PyTorch Dataset for the MBPP dataset. Uses only the sanitized
    version of the dataset.

    See: https://huggingface.co/datasets/mbpp
    """

    def __init__(self):
        mbpp = load_dataset("mbpp", name="sanitized")
        self.dataset = concatenate_datasets(
            [mbpp["train"], mbpp["test"], mbpp["validation"]]  # type: ignore
        )

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return self.dataset.num_rows

    def __getitem__(
        self, key: Union[int, Iterable[int], slice]
    ) -> Union[tuple[str, list[str]], tuple[str, list[str]]]:
        """
        Returns a tuple containing:
            prompt (str | list[str]): problem text prompt with test code
            test_code (str | list[str]): only test code, runnable with exec()
        """
        datasubset = self.dataset[key]

        if isinstance(key, int):
            prompt = prep_prompt(
                datasubset["prompt"],
                datasubset["test_imports"],
                datasubset["test_list"],
            )
            test_code = "\n".join(datasubset["test_imports"] + datasubset["test_list"])
        else:
            zipped_datasubset = zip(
                datasubset["prompt"],
                datasubset["test_imports"],
                datasubset["test_list"],
            )
            prompt, test_code = [], []
            for p, ti, tl in zipped_datasubset:
                prompt.append(prep_prompt(p, ti, tl))
                test_code.append("\n".join(ti + tl))

        return (prompt, test_code)  # type: ignore
