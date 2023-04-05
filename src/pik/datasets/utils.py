import random
import re
from typing import Optional, Union

import pandas as pd
from tqdm.auto import tqdm


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


def _chunked(indexable, n):
    """Yield successive n-sized chunks from an indexable object.
    Last chunk may be smaller than n.
    """
    for i in range(0, len(indexable), n):
        yield indexable[i : i + n]  # noqa


def get_token_seq_lens(
    dataset,
    tokenizer,
    data_ids: Optional[list[int]] = None,
    dataset_input_index: Optional[int] = 0,
    prompt_template: Optional[str] = None,
    use_tqdm: bool = False,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Returns a DataFrame containing token sequence lengths of each data id.

    Args:
        dataset: anything with a __len__ method implemented
        tokenizer: a tokenizer from transformers
        data_ids (list[int]): list of data ids to get token sequence lengths for
        dataset_input_index (int or None): index of the input in the dataset
            eg. input_text = dataset[i][dataset_input_index].
            Set to None if input_text = dataset[i] (no 2nd index needed)
        prompt_template (str): template for the prompt. eg. 'Q: {input} A:'
        use_tqdm (bool): whether to use tqdm to show progress
        chunk_size (int): number of data ids to process at once

    Returns:
        df (pd.DataFrame): DataFrame containing data_id and seq_len columns
            If prompt_template is not None, df will also contain a seq_len_with_template column
    """
    if data_ids is None:
        data_ids = list(range(len(dataset)))

    num_chunks = int((len(data_ids) / chunk_size).__ceil__())
    chunks = _chunked(data_ids, chunk_size)
    if use_tqdm:
        chunks = tqdm(chunks, total=num_chunks)

    df = pd.DataFrame(columns=["data_id", "seq_len"])
    for chunk in chunks:
        if dataset_input_index is not None:
            batch = [dataset[i][dataset_input_index] for i in chunk]
        else:
            batch = [dataset[i] for i in chunk]
        seq_lens = [len(ids) for ids in tokenizer(batch)["input_ids"]]

        new = pd.DataFrame()
        new["data_id"] = chunk
        new["seq_len"] = seq_lens
        df = pd.concat([df, new], ignore_index=True, axis=0)

    if prompt_template:
        i = data_ids[0]
        if dataset_input_index is not None:
            input = dataset[i][dataset_input_index]
        else:
            input = dataset[i]
        prompt = prompt_template.format(input=input)
        prompt_len = len(tokenizer(prompt)["input_ids"])
        input_len = len(tokenizer(input)["input_ids"])
        df["seq_len_with_template"] = df["seq_len"] + prompt_len - input_len

    return df
