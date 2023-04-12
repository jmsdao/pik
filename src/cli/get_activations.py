import argparse
import sys
from pathlib import Path
from functools import partial
from collections import defaultdict

import torch
import yaml

from pik.datasets import load_dataset
from pik.datasets.utils import get_data_ids, get_data_ids_from_file
from pik.models import load_model, load_tokenizer
from pik.models.hooks import HookedModule

from .helpers import (
    Timer, LINE_BREAK,
    validate_file, validate_local_dir,
    check_s3_write_access,
    check_files_exist_locally, check_files_exist_s3
)

torch.set_grad_enabled(False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config", type=str,
        help="Path to config file (.yaml). See cli/configs/ for examples",
    )
    parser.add_argument(
        "-e", "--estimate", action="store_true",
        help="Estimate the total disk space used by collected activations",
    )

    return parser.parse_args()


def validate_config(cli: str) -> None:
    """Validate config file."""
    # Future feature: validate config file against a schema
    if not cli:
        print('Error: config file does not specify "cli"')
        sys.exit()
    elif cli not in ["get_activations", "get-activations"]:
        print(f'Error: config file is for "{cli}"')
        sys.exit()


def append_postfix(filename, postfix):
    """Append a postfix to a filename"""
    return Path(filename).stem + postfix + Path(filename).suffix


def get_metadata_filenames(args: argparse.Namespace, config: dict) -> dict:
    """Get filenames for metadata files to save."""
    postfix = config["results"].get("postfix", "")

    filenames = {}
    filenames["config"] = Path(args.config).name
    filenames["repo_info"] = append_postfix("repo_info.txt", postfix)
    filenames["hardware_info"] = append_postfix("hardware_info.txt", postfix)
    filenames["environment"] = append_postfix("environment.yaml", postfix)

    # Sort the dict by value
    filenames = dict(sorted(filenames.items(), key=lambda item: item[1]))

    return filenames


def get_pickle_filenames(
    config: dict,
    messenger: dict,
    num_questions: int,
) -> list[str]:
    """Get filenames for pickle files to save activations to."""
    postfix = config.get("postfix", "")
    messenger_keys = sorted(list(messenger.keys()))
    file_prefixes = [key + postfix for key in messenger_keys]

    save_frequency = config["results"].get("save_frequency", None)
    if save_frequency:
        num_fileparts = int((num_questions / save_frequency).__ceil__())
        file_prefixes = [
            prefix + f"-{i+1 :05d}-of-{num_fileparts :05d}"
            for prefix in file_prefixes
            for i in range(num_fileparts)
        ]

    filenames = [prefix + ".pkl" for prefix in file_prefixes]

    return filenames


def get_resulting_filenames(
    args: argparse.Namespace,
    config: dict,
    messenger: dict,
    num_questions: int,
) -> dict:
    """Combine pickle and metadata filenames into a single list."""
    filenames = {}
    for fname in get_pickle_filenames(config, messenger, num_questions):
        filenames[fname] = fname

    filenames.update(get_metadata_filenames(args, config))

    return filenames


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh.read())

    validate_config(config.get("cli", None))

    # Validate ids_file path if specified
    ids_file = config["dataset"].get("ids_file", None)
    validate_file(ids_file)

    print(f"Running experiment using config {args.config}")

    # Check at least one of local_dir or s3_dir is specified
    if config["results"].get("dir", None) is None:
        print("Error: No local dir or s3 dir specified in config file")
        sys.exit()

    local_dir = config["results"]["dir"].get("local", None)
    s3_uri = config["results"]["dir"].get("s3", None)
    if not local_dir and not s3_uri:
        print("Error: No local dir or s3 dir specified in config file")
        sys.exit()

    # --- LOAD MODEL AND TOKENIZER --------------------------------------------
    model_name = config["model"]
    print(LINE_BREAK)
    print(f'Loading model "{model_name}"...')
    with Timer("Loaded model in {}\n"):
        model, tokenizer = load_model(model_name), load_tokenizer(model_name)
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
    print(f"model.__class__: {model.__class__}")
    print(f"model.dtype: {model.dtype}")
    print(f"model.device: {model.device}")
    if getattr(model, "hf_device_map", None):
        print(f"model.hf_device_map:{model.hf_device_map}")

    hooked_model = HookedModule(model)

    # --- LOAD DATASET AND EVALUATION FUNCTION --------------------------------
    dataset_name = config["dataset"]["name"]
    print(LINE_BREAK)
    print(f'Loading dataset "{dataset_name}"...')
    with Timer("Loaded dataset in {}"):
        dataset = load_dataset(dataset_name)

    # Get question ids of questions to process
    if ids_file:
        qids = get_data_ids_from_file(dataset, ids_file)
        num_questions = len(qids)
    else:
        qids = get_data_ids(
            dataset,
            num_items=config["dataset"]["num_questions"],
            skip=config["dataset"].get("skip", None),
            shuffle=config["dataset"]["shuffle"],
            shuffle_seed=config["dataset"].get("seed", None),
        )
        num_questions = len(qids)
    print(f"Number of questions to process: {num_questions}")

    # --- GET RESULTING FILENAMES ---------------------------------------------
    print(LINE_BREAK)
    print("Getting resulting filenames...")

    exec_globals = {}
    exec(config["hook_fns"], exec_globals)
    forward_kwargs = config.get("forward_kwargs", {})

    messenger = defaultdict(list)

    fwd_hook_fns = []
    for path in exec_globals["modules_to_hook"]:
        hook_fn = partial(
            exec_globals["fwd_hook_function"],
            module_path=path,
            messenger=messenger
        )
        fwd_hook_fns.append((path, hook_fn))

    text = "The content of this is irrelevant"
    encoded_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # type: ignore
    with torch.inference_mode(), hooked_model.hooks(fwd=fwd_hook_fns):
        hooked_model(**encoded_inputs, **forward_kwargs)

    resulting_filenames = get_resulting_filenames(args, config, messenger, num_questions)

    # Local file checks
    if local_dir:
        validate_local_dir(local_dir)
        if not config["results"]["overwrite"]:
            print("Checking if files already exist locally...")
            check_files_exist_locally(config, resulting_filenames)

    # S3 checks
    if s3_uri:
        if not config["results"]["overwrite"]:
            print("Checking if files already exist on S3...")
            check_files_exist_s3(config, resulting_filenames)
        print(f"Checking write access to {s3_uri}...")
        check_s3_write_access(args, s3_uri)
        print("Write access confirmed!")

    # --- RUN ESTIMATION ------------------------------------------------------
    raise NotImplementedError("WIP")


if __name__ == "__main__":
    main()
