from typing import Union, Optional
import random
import re


def get_data_ids(
    dataset,
    num_items: Union[int, str] = "all",
    skip: Optional[int] = None,
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
) -> list[int]:
    """Returns a list of data ids to use for training or evaluation.

    Args:
        dataset: anything with a __len__ method implemented
        num_items (int | str): number of data ids to return. Set to 'all' to return all data ids
        skip (int): number of data ids to skip. Will reduce num_items by the same amount
        shuffle (bool): whether to shuffle the data ids
        shuffle_seed (int): seed used for shuffling the data ids

    Returns:
        data_ids (list[int]): list of data ids
    """
    if isinstance(num_items, str) and num_items != "all":
        raise ValueError(f'num_items must be an int or "all", not "{num_items}"')
    if skip is not None and skip < 0:
        raise ValueError(f'skip must be >= 0, not "{skip}"')

    if num_items == "all" or num_items > len(dataset):
        num_items = len(dataset)

    if skip is not None and skip >= num_items:
        raise ValueError(f'skip must be < num_items={num_items}, not "{skip}"')

    if shuffle_seed is not None:
        random.seed(shuffle_seed)

    data_ids = list(range(len(dataset)))

    if shuffle:
        random.shuffle(data_ids)

    data_ids = data_ids[skip:num_items]

    return data_ids


def get_data_ids_from_file(dataset, filepath: str) -> list[int]:
    """Returns a list of data ids from a file.
    Splits on whitespace and commas.

    Args:
        dataset: anything with a __len__ method implemented
        filepath (str): path to file containing data ids

    Returns:
        data_ids (list[int]): list of data ids
    """
    with open(filepath, "r") as f:
        contents = f.read()
        data_ids = [int(i) for i in re.split(r"[\s,]+", contents) if i != ""]

    # Check all data ids are positive
    if any(i < 0 for i in data_ids):
        raise ValueError(f"Found a negative data id in file {filepath}")

    # Warn user if any ids are not unique
    non_unique = len(data_ids) - len(set(data_ids))
    if non_unique > 0:
        print(f"[ WARNING ] found {non_unique} non-unique data ids in file {filepath}")

    # Check all data ids are within the dataset
    if max(data_ids) >= len(dataset):
        raise ValueError(
            f"Found a data id ({max(data_ids)}) out of range of the dataset "
            f"(max_valid_id={len(dataset)-1}) in file {filepath}"
        )

    return data_ids
