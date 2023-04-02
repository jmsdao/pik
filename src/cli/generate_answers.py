import argparse
import shutil
import sys
from io import StringIO
from pathlib import Path
from time import time

import boto3
import pandas as pd
import yaml
from dotenv import dotenv_values
from tqdm import trange
from transformers import GenerationConfig

from pik import ROOT_DIR
from pik.datasets import load_dataset_and_eval_fn
from pik.datasets.utils import get_data_ids, get_data_ids_from_file
from pik.models import load_model_and_tokenizer
from pik.models.text_generation import TextGenerator

SECRETS = dotenv_values(Path(ROOT_DIR, ".env"))
N_TESTS = 3  # Number of questions to process when estimating


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config", type=str,
        help="Path to config file (.yaml). See cli/configs/ for examples.",
    )
    parser.add_argument(
        "-e", "--estimate", action="store_true",
        help="Estimate the total runtime of the script without running fully.",
    )

    return parser.parse_args()


def validate_config(cli: str) -> None:
    """Validate config file."""
    # Future feature: validate config file against a schema
    if not cli:
        print('Error: config file does not specify "cli"')
        sys.exit()
    elif cli not in ["generate_answers", "gen-answers"]:
        print(f'Error: config file is for "{cli}"')
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


def check_files_exist_locally(
    args: argparse.Namespace,
    config: dict,
) -> None:
    """Alert user if any local filepaths already exist."""
    local_dir = config["results"]["dir"]["local"]
    output_filepaths = [Path(local_dir) / Path(args.config).name]

    for filename in config["results"]["files"].values():
        output_filepaths.append(Path(local_dir) / filename)

    existing_paths = [path for path in output_filepaths if path.exists()]

    if existing_paths:
        print("Error: The following local filepaths already exist:")
        for path in existing_paths:
            print(f"  {path}")
        sys.exit()


def save_results_locally(
    args: argparse.Namespace,
    config: dict,
    text_gens: pd.DataFrame,
    qa_pairs: pd.DataFrame,
) -> None:
    """Save results locally."""
    local_dir = config["results"]["dir"]["local"]

    # Save config file to disk
    shutil.copy(args.config, Path(local_dir))

    # Save text generations to disk
    text_gens.to_csv(
        Path(local_dir) / config["results"]["files"]["text_generations"], index=False
    )

    # Save QA pairs to disk
    qa_pairs["qid"] = qa_pairs["qid"].astype(int)
    qa_pairs.to_csv(
        Path(local_dir) / config["results"]["files"]["qa_pairs"], index=False
    )


def validate_s3_uri(s3_uri: str) -> None:
    """Validate s3_uri in config file."""
    if s3_uri:
        if not s3_uri.startswith("s3://"):
            print(f'Error: s3_uri "{s3_uri}" must start with "s3://"')
            sys.exit()


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
    args: argparse.Namespace,
    config: dict,
) -> None:
    """Alert user if any s3 filepaths already exist."""
    s3_uri = config["results"]["dir"]["s3"]
    bucket_name, s3_key_prefix = get_s3_bucket_name_and_key(s3_uri)
    bucket = connect_to_s3(bucket_name)

    output_keys = [s3_key_prefix + Path(args.config).name]

    for filename in config["results"]["files"].values():
        output_keys.append(s3_key_prefix + filename)

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


def save_results_s3(
    args: argparse.Namespace,
    config: dict,
    text_gens: pd.DataFrame,
    qa_pairs: pd.DataFrame,
) -> None:
    """Save results to s3."""
    s3_uri = config["results"]["dir"]["s3"]
    bucket_name, s3_key_prefix = get_s3_bucket_name_and_key(s3_uri)
    bucket = connect_to_s3(bucket_name)

    config_key = s3_key_prefix + Path(args.config).name
    text_gens_key = s3_key_prefix + config["results"]["files"]["text_generations"]
    qa_pairs_key = s3_key_prefix + config["results"]["files"]["qa_pairs"]

    # Upload config file
    try:
        bucket.upload_file(args.config, config_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {args.config} to s3 {config_key}")
        print("Continuing...")

    # Upload text generations from IO buffer
    try:
        csv_buffer = StringIO()
        text_gens.to_csv(csv_buffer, index=False)
        bucket.put_object(Body=csv_buffer.getvalue(), Key=text_gens_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload text generations to s3 {text_gens_key}")
        print("Continuing...")

    # Upload qa pairs from IO buffer
    try:
        csv_buffer = StringIO()
        qa_pairs["qid"] = qa_pairs["qid"].astype(int)
        qa_pairs.to_csv(csv_buffer, index=False)
        bucket.put_object(Body=csv_buffer.getvalue(), Key=qa_pairs_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload QA pairs to s3 {qa_pairs_key}")
        print("Continuing...")


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh.read())

    validate_config(config.get("cli", None))

    # Validate ids_file path if specified
    ids_file = config["dataset"].get("ids_file", None)
    validate_file(ids_file)

    if args.estimate:
        print(f"Estimating runtime of full experiment from config {args.config}")
    else:
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

    # Local file checks
    if local_dir and not args.estimate:
        validate_local_dir(local_dir)
        if not config["results"]["overwrite"]:
            check_files_exist_locally(args, config)

    # S3 checks
    if s3_uri and not args.estimate:
        if not config["results"]["overwrite"]:
            check_files_exist_s3(args, config)
        print(f"Checking write access to {s3_uri}...")
        check_s3_write_access(args, s3_uri)
        print("Write access confirmed!")

    # Load model and tokenizer
    model_name = config["model"]
    print(f'Loading model "{model_name}"...')
    with Timer("Loaded model in {}"):
        model, tokenizer = load_model_and_tokenizer(model_name)

    text_generator = TextGenerator(
        model,
        tokenizer,
        gen_config=GenerationConfig(**config["generation"]["config"]),
        generation_seed=config["generation"].get("seed", None),
    )

    # Load dataset and its evaluation function
    dataset_name = config["dataset"]["name"]
    print(f'Loading dataset "{dataset_name}"...')
    with Timer("Loaded dataset in {}"):
        dataset, eval_fn = load_dataset_and_eval_fn(dataset_name)

    # Get question ids of questions to process
    if ids_file:
        qids = get_data_ids_from_file(dataset, ids_file)
        num_questions = len(qids)
    else:
        num_questions = config["dataset"]["num_questions"]
        qids = get_data_ids(
            dataset,
            num_items=num_questions,
            skip=config["dataset"]["skip"],
            shuffle=config["dataset"]["shuffle"],
            shuffle_seed=config["dataset"]["seed"],
        )

    # Grab repeated parameters from config
    generations_per_question = config["generation"]["generations_per_question"]
    batchsize_per_pass = config["generation"]["batchsize_per_pass"]
    max_new_tokens = config["generation"]["config"]["max_new_tokens"]

    # Estimate full runtime
    if args.estimate:
        start = time()
        progress_bar = trange(N_TESTS)
        for i in progress_bar:
            # Prep inputs
            question, answers = dataset[qids[i]]
            text_input = text_generator.prompt_engineer(
                config["prompt_template"], question
            )

            # Generate model ouputs and evaluate
            text_outputs = text_generator.generate(
                text_input,
                num_generations=generations_per_question,
                batchsize_per_pass=batchsize_per_pass,
            )
            evaluations = eval_fn(text_outputs, answers)

        time_taken = time() - start

        print("-" * 80)
        print(
            f"Estimation parameters:\n"
            f"Num questions processed = {N_TESTS}\n"
            f"Generations per question = {generations_per_question}\n"
            f"Max generations per forward pass = {batchsize_per_pass}\n"
            f"Max new tokens per generation = {max_new_tokens}\n"
            f"\n"
            f"Average processing time per question:\n"
            f"{time_taken / N_TESTS :.2f} seconds\n"
            f"\n"
            f"Estimated duration to process {num_questions} questions:\n"
            f"{(time_taken / N_TESTS) * num_questions / 3600 :.3f} hours\n\n"
            f"Note: does not include time taken to write to disk or upload to s3"
        )
        print("-" * 80)

        if local_dir:
            print("Output files to be saved to local disk:")
            for filename in config["results"]["files"].values():
                print(f"  {Path(local_dir) / filename}")
            print(f"  {Path(local_dir) / Path(args.config).name}")
            print("-" * 80)

        if s3_uri:
            if not s3_uri.endswith("/"):
                s3_uri += "/"
            print("Output files to be saved to s3:")
            for filename in config["results"]["files"].values():
                print(f"  {s3_uri}{filename}")
            print(f"  {s3_uri}{Path(args.config).name}")
            print("-" * 80)

        sys.exit()

    # Run the experiment
    text_gens = pd.DataFrame()
    qa_pairs = pd.DataFrame()

    save_frequency = config["results"].get("save_frequency", None)

    progress_bar = trange(num_questions)
    for i in progress_bar:
        # Prep inputs
        question, answers = dataset[qids[i]]
        text_input = text_generator.prompt_engineer(
            config["prompt_template"], question
        )

        # Generate model ouputs and evaluate
        text_outputs = text_generator.generate(
            text_input,
            num_generations=generations_per_question,
            batchsize_per_pass=batchsize_per_pass,
        )
        evaluations = eval_fn(text_outputs, answers)

        # Collect results
        qa_pairs.loc[i, "qid"] = qids[i]
        qa_pairs.loc[i, "question"] = question
        qa_pairs.loc[i, "answers"] = ";".join(answers)

        df = pd.DataFrame()
        df["qid"] = [qids[i]] * len(text_outputs)
        df["n"] = [n for n in range(1, len(text_outputs) + 1)]
        df["model_answer"] = text_outputs
        df["evaluation"] = evaluations
        text_gens = pd.concat([text_gens, df], ignore_index=True)

        # Update progress bar
        progress_bar.set_description(
            f'Last qid={qids[i]}, mean_eval={df["evaluation"].mean() :.2f}'
        )

        # Periodically save results
        if save_frequency and (i + 1) % save_frequency == 0:
            if local_dir:
                save_results_locally(args, config, text_gens, qa_pairs)
            if s3_uri:
                save_results_s3(args, config, text_gens, qa_pairs)

    # Save final results
    if local_dir:
        print(f"Saving final results to {local_dir}")
        save_results_locally(args, config, text_gens, qa_pairs)
    if s3_uri:
        print(f"Saving final results to {s3_uri}")
        save_results_s3(args, config, text_gens, qa_pairs)

    print("Success!")


if __name__ == "__main__":
    main()
