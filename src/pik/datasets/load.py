def load_dataset_and_eval_fn(dataset_name: str) -> tuple:
    """Syntactic sugar for loading a dataset and evaluation function."""
    # Conditional imports to avoid slow imports
    if dataset_name == 'trivia_qa':
        from .trivia_qa import TriviaQADataset, evaluate_answer  # pylint: disable=import-outside-toplevel
        return TriviaQADataset(), evaluate_answer

    raise ValueError(f'Unknown dataset "{dataset_name}"')
