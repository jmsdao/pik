from pathlib import Path


def _get_root(file_depth: int = 3) -> Path:
    """Returns the absolute path (Path) to the root directory of the repo.

    `file_depth`: The number of directories to go up from the current file to
    reach the root directory.

    DEV NOTE: this is pretty hacky, and will break if the file structure
    changes. Is there a better way to do this?
    """
    root = Path(__file__).resolve()
    for _ in range(file_depth):
        root = root.parent

    return root


ROOT_DIR = _get_root()
