from typing import Union, Optional
import torch


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
        skip (int): number of data ids to skip
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
        torch.manual_seed(shuffle_seed)

    if shuffle:
        data_ids = torch.randperm(len(dataset)).cpu().numpy().tolist()[skip:num_items]
    else:
        data_ids = list(range(num_items))[skip:]

    return data_ids
