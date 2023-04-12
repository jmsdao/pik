import argparse
import subprocess
import sys
from pathlib import Path
from time import time

import boto3
import GPUtil
from dotenv import dotenv_values
from git.repo import Repo

from pik import ROOT_DIR

SECRETS = dotenv_values(Path(ROOT_DIR, ".env"))
LINE_BREAK = "-" * 80


class Timer:
    """Context manager for timing code."""

    def __init__(self, print_template):
        self.print_template = print_template
        self.start = 0

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        time_taken = time() - self.start
        if time_taken < 60:
            print(self.print_template.format(f"{time_taken:.3f} seconds"))
        elif time_taken < 3600:
            print(self.print_template.format(f"{time_taken / 60:.3f} minutes"))
        else:
            print(self.print_template.format(f"{time_taken / 3600:.3f} hours"))


def get_repo_info() -> str:
    """Returns info about current repo."""
    repo = Repo(ROOT_DIR)
    repo_url_main = repo.git.remote("get-url", "origin")
    commit_hash = repo.git.rev_parse("HEAD")
    repo_url_commit = f"{repo_url_main}{commit_hash}".replace(".git", "/tree/")
    repo_id = f"git+{repo_url_main}@{commit_hash}"
    repo_info = f"{repo_url_commit}\n{repo_id}\n"

    return repo_info


def get_hardware_info() -> str:
    info = ""
    gpus = GPUtil.getGPUs()

    if gpus:
        # Get nvidia-smi info
        try:
            info += subprocess.Popen(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ).communicate()[0].decode("utf-8") + "\n"

            info += subprocess.Popen(
                ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ).communicate()[0].decode("utf-8") + "\n"

        except Exception:
            pass
    else:
        info = "No GPUs\n"

    return info


def append_postfix(filename, postfix):
    """Append a postfix to a filename"""
    return Path(filename).stem + postfix + Path(filename).suffix


def validate_file(file: str) -> None:
    """Validate file exists and is readable."""
    if file:
        path = Path(file)
        if not path.exists():
            print(f'Error: file "{path}" does not exist')
            sys.exit()
        if not path.is_file():
            print(f'Error: file "{path}" is not a file')
            sys.exit()
        if not path.open().readable():
            print(f'Error: file "{path}" is not readable')
            sys.exit()


def validate_local_dir(local_dir: str) -> None:
    """Validate local_dir in config file."""
    if local_dir:
        path = Path(local_dir)
        if not path.exists():
            print(f'Error: local_dir "{path}" does not exist')
            sys.exit()
        if not path.is_dir():
            print(f'Error: local_dir "{path}" is not a directory')
            sys.exit()
        try:
            path.joinpath("test.txt").touch()
            path.joinpath("test.txt").unlink()
        except PermissionError:
            print(f'Error: local_dir "{path}" is not writable')
            sys.exit()


def check_files_exist_locally(
    config: dict,
    filenames: dict,
) -> None:
    """Alert user if any local filepaths already exist."""
    local_dir = config["results"]["dir"]["local"]

    output_filepaths = [Path(local_dir) / filename for filename in filenames.values()]
    existing_paths = sorted([path for path in output_filepaths if path.exists()])

    if existing_paths:
        print("Error: The following local filepaths already exist:")
        for path in existing_paths:
            print(f"  {path}")
        sys.exit()


# def validate_s3_uri(s3_uri: str) -> None:
#     """Validate s3_uri in config file."""
#     if s3_uri:
#         if not s3_uri.startswith("s3://"):
#             print(f'Error: s3_uri "{s3_uri}" must start with "s3://"')
#             sys.exit()


def get_s3_bucket_name_and_key(s3_uri: str) -> tuple[str, str]:
    """Get s3 bucket and key from s3_uri."""
    if not s3_uri.endswith("/"):
        s3_uri += "/"
    bucket_name, s3_key_prefix = s3_uri.replace("s3://", "").split("/", maxsplit=1)
    return bucket_name, s3_key_prefix


def connect_to_s3(bucket_name: str):
    """Connect to s3."""
    bucket = boto3.resource(
        "s3",
        aws_access_key_id=SECRETS["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=SECRETS["AWS_SECRET_ACCESS_KEY"],
    ).Bucket(bucket_name)  # type: ignore

    return bucket


def check_files_exist_s3(
    config: dict,
    filenames: dict,
) -> None:
    """Alert user if any s3 filepaths already exist."""
    s3_uri = config["results"]["dir"]["s3"]
    bucket_name, s3_key_prefix = get_s3_bucket_name_and_key(s3_uri)
    bucket = connect_to_s3(bucket_name)

    output_keys = [s3_key_prefix + fname for fname in filenames.values()]

    existing_keys = []
    try:
        for obj in bucket.objects.filter(Prefix=s3_key_prefix):
            if obj.key in output_keys:
                existing_keys.append(obj.key)
    except Exception as e:
        print(e)
        print(f"Could not read from: {s3_uri}")
        sys.exit()

    if existing_keys:
        existing_keys = sorted(existing_keys)
        print("Error: The following s3 filepaths already exist:")
        for key in existing_keys:
            print(f"  s3://{bucket_name}/{key}")
        sys.exit()


def check_s3_write_access(
    args: argparse.Namespace,
    s3_uri: str,
) -> None:
    """Alert user if they don't have write access to s3."""
    bucket_name, s3_key_prefix = get_s3_bucket_name_and_key(s3_uri)
    bucket = connect_to_s3(bucket_name)
    s3_key = s3_key_prefix + Path(args.config).name

    try:
        bucket.upload_file(args.config, s3_key)
    except Exception as e:
        print(e)
        print(f"Could not write to: {s3_uri}")
        sys.exit()


def readable_size(size, decimal_point=3):
    suffix = "B"
    for i in ["B", "KB", "MB", "GB", "TB"]:
        suffix = i
        if size <= 1000.0:
            break
        size /= 1000.0

    return f"{size:.{decimal_point}f} {suffix}"
