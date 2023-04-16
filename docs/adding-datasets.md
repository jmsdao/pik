## Steps for adding a new dataset
1. Create a new directory in `pik/datasets/` with the name of the dataset
2. Implement a class for the new dataset in `pik/datasets/<dataset_name>/dataset.py`
    - It should inherit from `torch.utils.data.Dataset`
    - It should return a tuple of `(question, answer)` for each item
        - When indexed with an integer, it should return a single question/answer pair
        - When indexed with a slice or iterable, it should return a list of question/answer pairs
3. Implement an evaluation function in `pik/datasets/<dataset_name>/evaluate.py`
    - It should return a list of 1s and 0s indicating whether the model answer was correct for each model/dataset answer pair
4. In `pik/dataset/<dataset_name>/__init__.py`:
    - Import the dataset class and evaluation function
5. In `pik/datasets/load.py`:
    - Add the dataset name to the `IMPLEMENTED_DATASETS` list
    - Add an `if` block to load the dataset in the `load_dataset` function
    - Add an `if` block to load the evaluation function in the `get_eval_fn` function

See [`pik/datasets/trivia_qa/`](https://github.com/jmsdao/pik/tree/main/src/pik/datasets/trivia_qa) for an example.
