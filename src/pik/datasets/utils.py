from typing import Union, Optional
import torch


def get_data_ids(
        dataset,
        num_items: Union[int, str] = 'all',
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
    ) -> list[int]:
    """Returns a list of data ids to use for training or evaluation.
    
    Args:
        dataset: anything with a __len__ method implemented
        num_items (int | str): number of data ids to return. Set to 'all' to return all data ids
        shuffle (bool): whether to shuffle the data ids
        shuffle_seed (int): seed for random number generator

    Returns:
        data_ids (list[int]): list of data ids
    """
    if num_items != 'all' and isinstance(num_items, str):
        raise ValueError(f'num_items must be an int or "all", not "{num_items}"')

    if num_items == 'all' or num_items > len(dataset):
        num_items = len(dataset)

    if shuffle_seed is not None:
        torch.manual_seed(shuffle_seed)

    if shuffle:
        data_ids = torch.randperm(len(dataset))[:num_items].cpu().numpy().tolist()  # pylint: disable=no-member
    else:
        data_ids = list(range(num_items))

    return data_ids
