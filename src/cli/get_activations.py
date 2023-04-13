import argparse
import io
import pickle
import random
import shutil
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm, trange

from pik import ROOT_DIR
from pik.datasets import load_dataset
from pik.datasets.utils import get_data_ids, get_data_ids_from_file
from pik.models import load_model, load_tokenizer
from pik.models.hooks import HookedModule

from cli.helpers import (
    Timer, LINE_BREAK, append_postfix, readable_size,
    get_repo_info, get_hardware_info,
    validate_file, validate_local_dir,
    get_s3_bucket_name_and_key, connect_to_s3, check_s3_write_access,
    check_files_exist_locally, check_files_exist_s3
)

torch.set_grad_enabled(False)

N_TESTS = 5


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


def get_metadata_filenames(args: argparse.Namespace, config: dict) -> dict:
    """Get filenames for metadata files to save."""
    postfix = config["results"].get("postfix", "")

    filenames = {}
    filenames["config"] = Path(args.config).name
    filenames["repo_info"] = append_postfix("repo_info.txt", postfix)
    filenames["hardware_info"] = append_postfix("hardware_info.txt", postfix)
    filenames["environment"] = append_postfix("environment.yaml", postfix)
    filenames["file_indices"] = append_postfix("file_indices.csv", postfix)

    # Sort the dict by value
    filenames = dict(sorted(filenames.items(), key=lambda item: item[1]))

    return filenames


def get_pickle_filenames(
    config: dict,
    store: dict,
    num_questions: int,
) -> list[str]:
    """Get filenames for pickle files to save activations to."""
    postfix = config.get("postfix", "")
    store_keys = sorted(list(store.keys()))
    file_prefixes = [key + postfix for key in store_keys]

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
    store: dict,
    num_questions: int,
) -> dict:
    """Combine pickle and metadata filenames into a single list."""
    filenames = {}
    for fname in get_pickle_filenames(config, store, num_questions):
        filenames[fname] = fname

    filenames.update(get_metadata_filenames(args, config))

    return filenames


def save_results_locally(
    args: argparse.Namespace,
    config: dict,
    file_indices: pd.DataFrame,
    store: dict,
    cur_filepart: int,
    num_fileparts: int,
) -> None:
    """Save activations and metadata to local directory."""

    local_dir = config["results"]["dir"]["local"]
    filenames = get_metadata_filenames(args, config)

    # Save config file local_dir
    shutil.copy(args.config, Path(local_dir))

    # Save repo info local_dir
    with open(Path(local_dir) / filenames["repo_info"], "w") as f:
        f.write(get_repo_info())

    # Save hardware info local_dir
    with open(Path(local_dir) / filenames["hardware_info"], "w") as f:
        f.write(get_hardware_info())

    # Save environment.yaml to local_dir
    shutil.copy(
        ROOT_DIR / "environment.yaml",
        Path(local_dir) / filenames["environment"]
    )

    # Save file_indices.csv to local_dir
    file_indices["file_index"] = file_indices["file_index"].astype(int)
    file_indices["qid"] = file_indices["qid"].astype(int)
    file_indices.to_csv(
        Path(local_dir) / filenames["file_indices"], index=False,
    )

    # Save activations to local_dir
    for key, obj in store.items():
        fname = key
        if num_fileparts > 1:
            fname += f"-{cur_filepart :05d}-of-{num_fileparts :05d}"
        fname += ".pkl"
        with open(Path(local_dir) / fname, "wb") as f:
            pickle.dump(obj, f)


def save_results_s3(
    args: argparse.Namespace,
    config: dict,
    file_indices: pd.DataFrame,
    store: dict,
    cur_filepart: int,
    num_fileparts: int,
) -> None:
    """Save activations and metadata to S3."""
    filenames = get_metadata_filenames(args, config)

    s3_uri = config["results"]["dir"]["s3"]
    bucket_name, s3_key_prefix = get_s3_bucket_name_and_key(s3_uri)
    bucket = connect_to_s3(bucket_name)

    # Upload config file to S3
    config_key = s3_key_prefix + filenames["config"]
    try:
        bucket.upload_file(args.config, config_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {args.config} to s3 {config_key}")
        print("Continuing...")

    # Upload repo info to S3
    repo_info_key = s3_key_prefix + filenames["repo_info"]
    try:
        str_buffer = io.StringIO(get_repo_info())
        bucket.put_object(Body=str_buffer.getvalue(), Key=repo_info_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['repo_info']} to s3 {repo_info_key}")
        print("Continuing...")

    # Upload hardware info to S3
    hardware_info_key = s3_key_prefix + filenames["hardware_info"]
    try:
        str_buffer = io.StringIO(get_hardware_info())
        bucket.put_object(Body=str_buffer.getvalue(), Key=hardware_info_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['hardware_info']} to s3 {hardware_info_key}")
        print("Continuing...")

    # Upload environment.yaml to S3
    environment_key = s3_key_prefix + filenames["environment"]
    try:
        bucket.upload_file(ROOT_DIR / "environment.yaml", environment_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['environment']} to s3 {environment_key}")
        print("Continuing...")

    # Upload filesindices.csv to S3
    file_indices["file_index"] = file_indices["file_index"].astype(int)
    file_indices["qid"] = file_indices["qid"].astype(int)

    file_indices_key = s3_key_prefix + filenames["file_indices"]
    try:
        str_buffer = io.StringIO()
        file_indices.to_csv(str_buffer, index=False)
        bucket.put_object(Body=str_buffer.getvalue(), Key=file_indices_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['file_indices']} to s3 {file_indices_key}")
        print("Continuing...")

    # Upload activations to S3
    for key, obj in store.items():
        fname = key
        if num_fileparts > 1:
            fname += f"-{cur_filepart :05d}-of-{num_fileparts :05d}"
        fname += ".pkl"
        s3_key = s3_key_prefix + fname
        try:
            pkl_obj = pickle.dumps(obj)
            pkl_buffer = io.BytesIO(pkl_obj)
            pbar = tqdm(total=len(pkl_obj), unit="B", unit_scale=True, leave=True)
            pbar.set_description(f"(S3) Uploading {s3_key}")
            bucket.upload_fileobj(
                Fileobj=pkl_buffer,
                Key=s3_key,
                Callback=lambda bytes_uploaded: pbar.update(bytes_uploaded),
            )
            pbar.close()
        except Exception as e:
            print(e)
            print(f"Warning: failed to upload {fname} to s3 {s3_key}")
            print("Continuing...")


# --- MAIN --------------------------------------------------------------------
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

    # --- INIT CONFIG PARAMS AND OTHER SETUP ----------------------------------

    # Get params needed for a forward pass from config
    exec_globals = {}
    exec(config["hook_fns"], exec_globals)
    modules_to_hook = exec_globals["modules_to_hook"]
    fwd_hook_function = exec_globals["fwd_hook_function"]
    forward_kwargs = config.get("forward_kwargs", {})

    # Calculate file related params
    save_frequency = config["results"].get("save_frequency", None)
    if save_frequency:
        num_fileparts = int((num_questions / save_frequency).__ceil__())
    else:
        num_fileparts = 1

    # Setup activations store and hook functions
    store = defaultdict(list)
    fwd_hook_fns = []
    for path in modules_to_hook:
        hook_fn = partial(fwd_hook_function, module_path=path, store=store)
        fwd_hook_fns.append((path, hook_fn))

    # Run a forward pass to get the activations
    text = "The content of this is irrelevant"
    encoded_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # type: ignore
    with torch.inference_mode(), hooked_model.hooks(fwd=fwd_hook_fns):
        hooked_model(**encoded_inputs, **forward_kwargs)

    resulting_filenames = get_resulting_filenames(args, config, store, num_questions)

    # Clear store
    store.clear()

    # --- LOCAL AND S3 FILE CHECKS --------------------------------------------
    if not args.estimate:
        print(LINE_BREAK)
        # Local file checks
        if local_dir and not args.estimate:
            validate_local_dir(local_dir)
            if not config["results"]["overwrite"]:
                print("Checking if files already exist locally...")
                check_files_exist_locally(config, resulting_filenames)

        # S3 checks
        if s3_uri and not args.estimate:
            if not config["results"]["overwrite"]:
                print("Checking if files already exist on S3...")
                check_files_exist_s3(config, resulting_filenames)
            print(f"Checking write access to {s3_uri}...")
            check_s3_write_access(args, s3_uri)
            print("Write access confirmed!")

    # --- RUN ESTIMATION ------------------------------------------------------
    if args.estimate:
        print(LINE_BREAK)
        print(f"Running file size estimation using {N_TESTS} randomly sampled qids...")

        # Run forward passes to collect activations
        sampled_qids = random.choices(qids, k=N_TESTS)
        for qid in tqdm(sampled_qids):
            question, _ = dataset[qid]
            text_input = config["prompt_template"].format(question)
            encoded_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)  # type: ignore
            with torch.inference_mode(), hooked_model.hooks(fwd=fwd_hook_fns):
                hooked_model(**encoded_inputs, **forward_kwargs)
            # Change last `hook_obj` to `(qid, hook_obj)` for each key in store
            for key in store.keys():
                store[key][-1] = (qid, store[key][-1])

        # Get test sizes, in bytes
        test_sizes = {}
        for key, obj in store.items():
            test_sizes[key] = len(pickle.dumps(obj))  # type: ignore

        # Print size estimation results
        total_size = 0
        print(f"\nEstimated files sizes for all {num_questions} questions:")
        for key, size in test_sizes.items():
            full_size = size * num_questions / N_TESTS
            total_size += full_size
            line_item = f"  {key}: {readable_size(full_size)}"
            if save_frequency and num_fileparts > 1:
                split_size = size * save_frequency / N_TESTS
                line_item += (
                    f" (split into {num_fileparts} parts with max filesize"
                    f" approx. {readable_size(split_size)})"
                )
            print(line_item)

        print(f"\nTotal estimated size: {readable_size(total_size)}")

        # List out files to be saved to local disk and S3
        print(LINE_BREAK)
        sorted_filenames = sorted(resulting_filenames.values())
        if local_dir:
            print("Output files to be saved to local disk:")
            for fname in sorted_filenames:
                print(f"  {Path(local_dir) / fname}")
            print()

        if s3_uri:
            if not s3_uri.endswith("/"):
                s3_uri += "/"
            print("Output files to be saved to s3:")
            for fname in sorted_filenames:
                print(f"  {s3_uri}{fname}")
            print()

        if config["results"]["overwrite"]:
            print("WARNING: Overwrite flag is set to True.")
            print("Make sure you actually want to overwrite existing files!")
        print(LINE_BREAK)

        sys.exit()

    # --- RUN ACTIVATION COLLECTION -------------------------------------------
    print(LINE_BREAK)
    print("Collecting activations...")
    if save_frequency:
        print(f"Saving activations every {save_frequency} questions")

    file_indices = pd.DataFrame()
    cur_filepart = 1
    fid = 0

    progress_bar = trange(len(qids))
    for i in progress_bar:
        save_this_iter = save_frequency and (i + 1) % save_frequency == 0

        # Setup input
        qid = qids[i]
        question, _ = dataset[qid]
        text_input = config["prompt_template"].format(question)
        encoded_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)  # type: ignore

        # Update progress bar
        bar_desc = f"(get-activations) Processing qid={qid}"
        if save_this_iter:
            bar_desc += " (saving...)"
        progress_bar.set_description(bar_desc)

        # Run forward pass
        with torch.inference_mode(), hooked_model.hooks(fwd=fwd_hook_fns):
            hooked_model(**encoded_inputs, **forward_kwargs)

        # Change last `hook_obj` to `(qid, hook_obj)` for each key in store
        for key in store.keys():
            store[key][-1] = (qid, store[key][-1])

        # Update file indices
        file_indices.loc[i, "qid"] = qid
        file_indices.loc[i, "file_index"] = fid
        if num_fileparts > 1:
            filepart_string = f"{cur_filepart :05d}-of-{num_fileparts :05d}"
            file_indices.loc[i, "filepart"] = filepart_string
        fid += 1

        # Periodically save activations to disk/S3
        if save_this_iter:
            if local_dir:
                save_results_locally(
                    args, config,
                    file_indices, store,
                    cur_filepart, num_fileparts,
                )
            if s3_uri:
                save_results_s3(
                    args, config,
                    file_indices, store,
                    cur_filepart, num_fileparts,
                )
            store.clear()
            cur_filepart += 1
            fid = 0

    # --- SAVE FINAL RESULTS --------------------------------------------------
    print(LINE_BREAK)
    # Might need extra conditionals for multi-part files
    if local_dir:
        print(f"Saving final results to {local_dir}")
        save_results_locally(
            args, config,
            file_indices, store,
            cur_filepart, num_fileparts,
        )
    if s3_uri:
        print(f"Saving final results to {s3_uri}")
        save_results_s3(
            args, config,
            file_indices, store,
            cur_filepart, num_fileparts,
        )

    print("Success!")


if __name__ == "__main__":
    main()
