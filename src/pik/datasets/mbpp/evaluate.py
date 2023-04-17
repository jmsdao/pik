from typing import Union


SOLUTIION_DELIM = "### START SOLUTION CODE"


def test_solution_code(solution: str, test: str) -> int:
    """Singular of the solution code against the test code. Success (1) is when
    the code runs without error.
    """
    code = f"{solution}\n{test}"
    try:
        # Empty dict for globals to prevent code from modifying the environment
        exec(code, {})
    except Exception:
        return 0
    return 1


def evaluate_answer(
    model_completions: Union[str, list[str]],
    dataset_test_code: Union[str, list[str]],
) -> list[int]:
    """Evaluate if model code completions are correct.

    Args:
        model_completions (str or list[str]): model code completions
        dataset_test_code (str or list[str]): test code from the dataset

    Returns:
        results (list[int]): 1 if model code completion passes tests, 0 otherwise
    """
    if isinstance(model_completions, str) and not isinstance(dataset_test_code, str):
        raise ValueError("If model_completions is str, dataset_test_code expects str.")
    if (isinstance(model_completions, list) and isinstance(model_completions[0], str)) and not (
        isinstance(dataset_test_code, list) and isinstance(dataset_test_code[0], str)
    ):
        raise ValueError(
            "If model_completions is list[str], dataset_test_code expects list[str]."
        )
    if isinstance(model_completions, list) and len(model_completions) != len(dataset_test_code):
        raise ValueError(
            "If model_completions is list[str], then len(model_completions) should equal len(dataset_test_code)."
        )

    if isinstance(model_completions, str):
        model_completions = [model_completions]
        dataset_test_code = [dataset_test_code]  # type: ignore

    results = []
    for completion, test in zip(model_completions, dataset_test_code):
        solution = completion.split(SOLUTIION_DELIM)[-1]
        results.append(test_solution_code(solution, test))

    return results
