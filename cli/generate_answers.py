import sys
from time import time
import shutil
from pathlib import Path
import argparse
import yaml

import pandas as pd
from tqdm import trange
from transformers import GenerationConfig

from pik.models import load_model_and_tokenizer
from pik.models.text_generation import TextGenerator
from pik.datasets import load_dataset_and_eval_fn
from pik.datasets.utils import get_data_ids


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
    elif cli != "generate_answers":
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


def check_local_paths_exists(filepaths: list[Path]) -> None:
    """Alert user if any local filepaths already exist."""
    existing_paths = [path for path in filepaths if path.exists()]
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

    shutil.copy(args.config, Path(local_dir))
    text_gens.to_csv(
        Path(local_dir) / config["results"]["files"]["text_generations"], index=False
    )
    qa_pairs["qid"] = qa_pairs["qid"].astype(int)
    qa_pairs.to_csv(
        Path(local_dir) / config["results"]["files"]["qa_pairs"], index=False
    )


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh.read())

    validate_config(config.get("cli", None))

    if args.estimate:
        print("Estimating runtime...")
    else:
        print("Running experiment...")

    # local_dir checks
    local_dir = config["results"]["dir"].get("local", None)
    if local_dir and not args.estimate:
        validate_local_dir(local_dir)
        output_filepaths = [Path(local_dir) / Path(args.config).name]
        for filename in config["results"]["files"].values():
            output_filepaths.append(Path(local_dir) / filename)
        check_local_paths_exists(output_filepaths)

    # s3_dir checks # TODO: implement

    # Check at least one of local_dir or s3_dir is specified
    if not local_dir:  # and not s3_dir:
        print("Error: No local dir or s3 dir specified in config file")
        sys.exit()

    # Load model and tokenizer
    model_name = config["model"]
    print(f'Loading model "{model_name}"...')
    with Timer("Loaded model in {}"):
        model, tokenizer = load_model_and_tokenizer(model_name)
    text_generator = TextGenerator(
        model,
        tokenizer,
        gen_config=GenerationConfig(**config["generation"]["config"]),
        generation_seed=config["generation"]["seed"],
    )

    # Load dataset and its evaluation function
    dataset_name = config["dataset"]["name"]
    print(f'Loading dataset "{dataset_name}"...')
    with Timer("Loaded dataset in {}"):
        dataset, eval_fn = load_dataset_and_eval_fn(dataset_name)

    # Get data ids of questions to process
    num_questions = config["dataset"]["num_questions"]
    qids = get_data_ids(
        dataset,
        num_items=num_questions,
        shuffle=config["dataset"]["shuffle"],
        shuffle_seed=config["dataset"]["seed"],
    )

    # Grab  repeated parameters from config
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
            evaluations = [eval_fn(output, answers) for output in text_outputs]

        time_taken = time() - start

        print(
            f"--------------------------------------------------------------------\n"
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
            f"Note: does not include time taken to write to disk or upload to s3\n"
            f"--------------------------------------------------------------------"
        )

        if local_dir:
            print("Output files to be saved to local disk:")
            for filename in config["results"]["files"].values():
                print(f"  {Path(local_dir) / filename}")
            print(f"  {Path(local_dir) / Path(args.config).name}")
            print(
                "--------------------------------------------------------------------"
            )

        sys.exit()

    # Run the experiment
    text_gens = pd.DataFrame()
    qa_pairs = pd.DataFrame()

    progress_bar = trange(num_questions)
    for i in progress_bar:
        # Prep inputs
        question, answers = dataset[qids[i]]
        text_input = text_generator.prompt_engineer(config["prompt_template"], question)

        # Generate model ouputs and evaluate
        text_outputs = text_generator.generate(
            text_input,
            num_generations=generations_per_question,
            batchsize_per_pass=batchsize_per_pass,
        )
        evaluations = [eval_fn(output, answers) for output in text_outputs]

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

        # Periodically save results
        if (i + 1) % config["results"]["save_frequency"] == 0:
            if local_dir:
                save_results_locally(args, config, text_gens, qa_pairs)

    # Save final results
    if local_dir:
        print(f"Saving final results to {local_dir}")
        save_results_locally(args, config, text_gens, qa_pairs)
