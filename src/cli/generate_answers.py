import argparse
import shutil
import sys
from io import StringIO
from pathlib import Path
from time import time

import GPUtil
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm, trange
from transformers import GenerationConfig

from pik import ROOT_DIR
from pik.datasets import get_eval_fn, load_dataset
from pik.datasets.utils import get_data_ids, get_data_ids_from_file, get_token_seq_lens
from pik.models import load_model, load_tokenizer
from pik.models.text_generation import TextGenerator

from .helpers import (
    Timer, LINE_BREAK, append_postfix,
    get_repo_info, get_hardware_info,
    validate_file, validate_local_dir,
    get_s3_bucket_name_and_key, connect_to_s3, check_s3_write_access,
    check_files_exist_locally, check_files_exist_s3
)

torch.set_grad_enabled(False)

N_TESTS_MEMORY = 3  # Number of passes with full batch size
N_TESTS_TIME = 5  # Number of questions to process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config", type=str,
        help="Path to config file (.yaml). See cli/configs/ for examples",
    )
    parser.add_argument(
        "-e", "--estimate", action="store_true",
        help="Estimate the total runtime of the script without running fully",
    )
    parser.add_argument(
        "-s", "--split-ids", metavar="SEQ_LEN", type=int, nargs='?',
        help=(
            "Split data ids from a given config by the given sequence length, and "
            "save the ids into two separate files (does not run experiment)"
        )
    )

    return parser.parse_args()


def gpu_usage():
    """Print GPU usage."""
    for i, gpu in enumerate(GPUtil.getGPUs()):
        print(f"GPU {i}:  {int(gpu.memoryUsed)} / {int(gpu.memoryTotal)}", end="")
        print(f" MB  ({(gpu.memoryUsed / gpu.memoryTotal) * 100 :.2f}%)")


def validate_config(cli: str) -> None:
    """Validate config file."""
    # Future feature: validate config file against a schema
    if not cli:
        print('Error: config file does not specify "cli"')
        sys.exit()
    elif cli not in ["generate_answers", "gen-answers"]:
        print(f'Error: config file is for "{cli}"')
        sys.exit()


def get_resulting_filenames(args: argparse.Namespace, config: dict) -> dict:
    postfix = config["results"].get("postfix", "")

    filenames = {}
    for key, name in config["results"]["files"].items():
        filenames[key] = append_postfix(name, postfix)

    filenames["config"] = Path(args.config).name
    filenames["repo_info"] = append_postfix("repo_info.txt", postfix)
    filenames["hardware_info"] = append_postfix("hardware_info.txt", postfix)
    filenames["environment"] = append_postfix("environment.yaml", postfix)

    # Sort the dict by value
    filenames = dict(sorted(filenames.items(), key=lambda item: item[1]))

    return filenames


def save_results_locally(
    args: argparse.Namespace,
    config: dict,
    text_gens: pd.DataFrame,
    qa_pairs: pd.DataFrame,
) -> None:
    """Save results locally."""
    local_dir = config["results"]["dir"]["local"]
    filenames = get_resulting_filenames(args, config)

    # Save config file local_dir
    shutil.copy(args.config, Path(local_dir))

    # Save text generations local_dir
    text_gens.to_csv(Path(local_dir) / filenames["text_generations"], index=False)

    # Save QA pairs local_dir
    qa_pairs["qid"] = qa_pairs["qid"].astype(int)
    qa_pairs.to_csv(Path(local_dir) / filenames["qa_pairs"], index=False)

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


def save_results_s3(
    args: argparse.Namespace,
    config: dict,
    text_gens: pd.DataFrame,
    qa_pairs: pd.DataFrame,
) -> None:
    """Save results to s3."""
    filenames = get_resulting_filenames(args, config)

    s3_uri = config["results"]["dir"]["s3"]
    bucket_name, s3_key_prefix = get_s3_bucket_name_and_key(s3_uri)
    bucket = connect_to_s3(bucket_name)

    config_key = s3_key_prefix + Path(args.config).name
    text_gens_key = s3_key_prefix + filenames["text_generations"]
    qa_pairs_key = s3_key_prefix + filenames["qa_pairs"]

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

    # Upload repo info from IO buffer
    repo_info = get_repo_info()
    repo_info_key = s3_key_prefix + filenames["repo_info"]
    try:
        csv_buffer = StringIO()
        csv_buffer.write(repo_info)
        bucket.put_object(Body=csv_buffer.getvalue(), Key=repo_info_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['repo_info']} to s3 {repo_info_key}")
        print("Continuing...")

    # Upload hardware info from IO buffer
    hardware_info = get_hardware_info()
    hardware_info_key = s3_key_prefix + filenames["hardware_info"]
    try:
        csv_buffer = StringIO()
        csv_buffer.write(hardware_info)
        bucket.put_object(Body=csv_buffer.getvalue(), Key=hardware_info_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['hardware_info']} to s3 {hardware_info_key}")
        print("Continuing...")

    # Upload environment.yaml
    environment_key = s3_key_prefix + filenames["environment"]
    try:
        bucket.upload_file(str(ROOT_DIR / "environment.yaml"), environment_key)
    except Exception as e:
        print(e)
        print(f"Warning: failed to upload {filenames['environment']} to s3 {environment_key}")
        print("Continuing...")


def generate_ids_files(config: dict, seq_len: int) -> None:
    dataset = load_dataset(config["dataset"]["name"])
    tokenizer = load_tokenizer(config["model"])
    ids_file = config["dataset"].get("ids_file", None)

    # Get question ids of questions to process
    if ids_file:
        qids = get_data_ids_from_file(dataset, ids_file)
    else:
        qids = get_data_ids(
            dataset,
            num_items=config["dataset"]["num_questions"],
            skip=config["dataset"].get("skip", None),
            shuffle=config["dataset"]["shuffle"],
            shuffle_seed=config["dataset"].get("seed", None),
        )

    df = get_token_seq_lens(
        dataset, tokenizer, qids,
        prompt_template=config["prompt_template"],
        use_tqdm=True
    )

    df_le = df[df["seq_len_with_template"] <= seq_len][["data_id"]]
    df_gt = df[df["seq_len_with_template"] > seq_len][["data_id"]]

    df_le.to_csv(f"ids_le{seq_len}.txt", index=False, header=False, sep="\n")
    df_gt.to_csv(f"ids_gt{seq_len}.txt", index=False, header=False, sep="\n")

    print(f"ids_le{seq_len}.txt: contains {len(df_le)} ids that are <= {seq_len}")
    print(f"ids_gt{seq_len}.txt: contains {len(df_gt)} ids that are > {seq_len}")

    sys.exit()


# --- MAIN --------------------------------------------------------------------
def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh.read())

    validate_config(config.get("cli", None))

    # Validate ids_file path if specified
    ids_file = config["dataset"].get("ids_file", None)
    validate_file(ids_file)

    if args.split_ids:
        generate_ids_files(config, args.split_ids)
        sys.exit()

    filenames = get_resulting_filenames(args, config)

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
            resulting_filenames = get_resulting_filenames(args, config)
            check_files_exist_locally(config, resulting_filenames)

    # S3 checks
    if s3_uri and not args.estimate:
        if not config["results"]["overwrite"]:
            resulting_filenames = get_resulting_filenames(args, config)
            check_files_exist_s3(config, resulting_filenames)
        print(f"Checking write access to {s3_uri}...")
        check_s3_write_access(args, s3_uri)
        print("Write access confirmed!")

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

    text_generator = TextGenerator(
        model,
        tokenizer,
        gen_config=GenerationConfig(**config["generation"]["config"]),
    )

    # --- LOAD DATASET AND EVALUATION FUNCTION --------------------------------
    dataset_name = config["dataset"]["name"]
    print(LINE_BREAK)
    print(f'Loading dataset "{dataset_name}"...')
    with Timer("Loaded dataset in {}"):
        dataset, eval_fn = load_dataset(dataset_name), get_eval_fn(dataset_name)

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

    # Grab repeated parameters from config
    generations_per_question = config["generation"]["generations_per_question"]
    batch_size = config["generation"]["batch_size"]
    max_new_tokens = config["generation"]["config"]["max_new_tokens"]

    # --- RUN ESTIMATION ------------------------------------------------------
    if args.estimate:
        print(LINE_BREAK)
        print("Finding input sequence length distribution (num tokens)...")

        df = get_token_seq_lens(
            dataset, tokenizer, qids,
            prompt_template=config["prompt_template"],
            use_tqdm=True,
        )

        median_seq_len = df["seq_len_with_template"].median()
        idx_median = (df["seq_len_with_template"] - median_seq_len).abs().idxmin()
        data_id_median = int(df.loc[idx_median, "data_id"])  # type: ignore

        max_seq_len = int(df["seq_len_with_template"].max())
        idx_max = int(df["seq_len_with_template"].idxmax())
        data_id_max = int(df.loc[idx_max, "data_id"])  # type: ignore

        # Get deciles
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        deciles = df["seq_len_with_template"].quantile(bins).round(1).tolist()
        print(f"\nDeciles: {deciles}")
        print(f"Median sequence length: {median_seq_len} (closest data_id={data_id_median})")
        print(f"Max sequence length: {max_seq_len} (data_id={data_id_max})")

        # Estimate GPU memory usage
        if GPUtil.getGPUs():
            print(LINE_BREAK)
            for test in ["median", "max"]:
                data_id = data_id_median if test == "median" else data_id_max
                print(f"Estimating {test} GPU memory usage using data_id={data_id}...")

                question, _ = dataset[data_id]  # type: ignore
                text_input = text_generator.prompt_engineer(
                    config["prompt_template"], question
                )
                text_inputs = [text_input] * batch_size

                # Run median memory test and print results
                for _ in trange(N_TESTS_MEMORY):
                    text_generator.generate(text_inputs)
                print(f"\nGPU memory usage ({test}):")
                gpu_usage()
                torch.cuda.empty_cache()
                if test == "median":
                    print()

        # Estimate time taken
        print(LINE_BREAK)
        print(f"Estimating generation time using data_id={data_id_median} (median)...")

        batched_qids = text_generator.get_batched_data_ids(
            [data_id_median] * N_TESTS_TIME, generations_per_question, batch_size  # type: ignore
        )

        bar_format = (
            "{desc}{percentage:.1f}%|{bar}| {n:.1f}/{total_fmt} "
            "[{elapsed}<{remaining},{rate_fmt}{postfix}]"
        )
        progress_bar = tqdm(total=N_TESTS_TIME, bar_format=bar_format)
        start = time()
        for batch in batched_qids:
            # Prep inputs
            if isinstance(batch, int):
                batch = [batch]
            questions, answers = dataset[batch]
            text_inputs = text_generator.prompt_engineer(
                config["prompt_template"], questions
            )

            # Generate model ouputs and evaluate
            text_outputs = text_generator.generate(text_inputs)
            evaluations = eval_fn(text_outputs, answers)

            # Update progress bar
            questions_in_batch = len(batch) / generations_per_question
            progress_bar.update(questions_in_batch)

        progress_bar.close()
        time_taken = time() - start

        # Print time estimation results
        print(
            f"\nTime estimation parameters:\n"
            f"Num questions processed = {N_TESTS_TIME}\n"
            f"Generations per question = {generations_per_question}\n"
            f"Max batch size = {batch_size}\n"
            f"Max new tokens per generation = {max_new_tokens}\n"
            f"\n"
            f"Average processing time per question:\n"
            f"{time_taken / N_TESTS_TIME :.2f} seconds\n"
            f"\n"
            f"Estimated duration to process {num_questions} questions:\n"
            f"{(time_taken / N_TESTS_TIME) * num_questions / 3600 :.3f} hours\n\n"
            f"Note: does not include time taken to write to disk or upload to s3"
        )

        # List out files to be saved to local disk and s3
        print(LINE_BREAK)
        if local_dir:
            print("Output files to be saved to local disk:")
            for fname in filenames.values():
                print(f"  {Path(local_dir) / fname}")
            print()

        if s3_uri:
            if not s3_uri.endswith("/"):
                s3_uri += "/"
            print("Output files to be saved to s3:")
            for fname in filenames.values():
                print(f"  {s3_uri}{fname}")
            print()

        if config["results"]["overwrite"]:
            print("WARNING: Overwrite flag is set to True.")
            print("Make sure you actually want to overwrite existing files!")
        print(LINE_BREAK)

        sys.exit()

    # --- RUN EXPERIMENT ------------------------------------------------------
    if config["generation"].get("seed", None):
        torch.manual_seed(config["generation"]["seed"])
    save_frequency = config["results"].get("save_frequency", None)
    num_questions_processed = 0

    print(LINE_BREAK)
    print("Generating text outputs...")
    if save_frequency:
        print(f"Saving results every {save_frequency} questions")

    text_gens = pd.DataFrame(columns=["qid", "model_answer", "evaluation"])
    qa_pairs = pd.DataFrame(columns=["qid", "question", "answer"])

    batched_qids = text_generator.get_batched_data_ids(
        qids, generations_per_question, batch_size
    )

    bar_format = (
        "{desc}{percentage:.1f}%|{bar}| {n:.1f}/{total_fmt} "
        "[{elapsed}<{remaining},{rate_fmt}{postfix}]"
    )
    progress_bar = tqdm(total=num_questions, bar_format=bar_format)
    for batch in batched_qids:
        # Prep inputs
        if isinstance(batch, int):
            batch = [batch]
        questions, answers = dataset[batch]
        text_inputs = text_generator.prompt_engineer(
            config["prompt_template"], questions
        )

        # Generate model ouputs and evaluate
        text_outputs = text_generator.generate(text_inputs)
        evaluations = eval_fn(text_outputs, answers)

        # Collect results
        batch_tg = pd.DataFrame()
        batch_tg["qid"] = batch
        batch_tg["model_answer"] = text_outputs
        batch_tg["model_answer"] = batch_tg["model_answer"].str.rstrip()
        batch_tg["evaluation"] = evaluations
        text_gens = pd.concat([text_gens, batch_tg], ignore_index=True)

        qids_in_batch = pd.Series(batch).unique().tolist()
        batch_qa = pd.DataFrame()
        new_qids = [i for i in qids_in_batch if i not in qa_pairs["qid"].tolist()]
        if new_qids:
            batch_qa["qid"] = new_qids
            batch_qa["question"] = batch_qa["qid"].apply(lambda x: dataset[x][0])
            batch_qa["answer"] = batch_qa["qid"].apply(lambda x: ";".join(dataset[x][1]))  # type: ignore
            qa_pairs = pd.concat([qa_pairs, batch_qa], ignore_index=True)

        questions_in_batch = len(batch) / generations_per_question
        num_questions_processed += questions_in_batch

        # Update progress bar
        mean_evals_per_qid = (
            batch_tg[["qid", "evaluation"]]
            .groupby("qid").mean().round(3)
            ["evaluation"].tolist()
        )
        bar_desc = (
            f"qids in batch: {qids_in_batch}, "
            f"mean evals per qid: {mean_evals_per_qid}"
        )
        if save_frequency and num_questions_processed >= save_frequency:
            bar_desc += " (saving results...)"
        progress_bar.update(questions_in_batch)
        progress_bar.set_description(bar_desc)

        # Periodically save results
        if save_frequency and num_questions_processed >= save_frequency:
            num_questions_processed = 0
            if local_dir:
                save_results_locally(args, config, text_gens, qa_pairs)
            if s3_uri:
                save_results_s3(args, config, text_gens, qa_pairs)

    progress_bar.close()

    # --- SAVE FINAL RESULTS --------------------------------------------------
    print(LINE_BREAK)
    if local_dir:
        print(f"Saving final results to {local_dir}")
        save_results_locally(args, config, text_gens, qa_pairs)
    if s3_uri:
        print(f"Saving final results to {s3_uri}")
        save_results_s3(args, config, text_gens, qa_pairs)

    print("Success!")


if __name__ == "__main__":
    main()
