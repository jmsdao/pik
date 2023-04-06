# This script uses conditional imports to minimize slow imports.
SMALL_MODELS = ["gpt2"]
LARGE_MODELS = ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]
IMPLEMENTED_MODELS = SMALL_MODELS + LARGE_MODELS


def load_model(model_name: str):
    """One-liner for loading models from the HuggingFace model hub.

    Args:
        model_name (str): Name of the model to load. Must be one of:
            ['test', 'gpt2', 'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']

    Returns:
        model: Model loaded from the HuggingFace model hub

    For testing, use model_str='test' to load gpt2 without downloading model
    weights from the HuggingFace model hub (weights will be random).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If testing, load a small model without downloading files
    if model_name == "test":
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_config(config).to(device)

    # Code block for loading small models
    elif model_name in SMALL_MODELS:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Code block for loading llama models
    elif "llama" in model_name:
        from transformers import LlamaForCausalLM
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from huggingface_hub import snapshot_download

        checkpoint_location = snapshot_download(f"decapoda-research/{model_name}-hf")

        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(checkpoint_location)

        model = load_checkpoint_and_dispatch(
            model,  # type: ignore
            checkpoint_location,
            device_map="auto",
            dtype=torch.float16,
            no_split_module_classes=["LlamaDecoderLayer"],
        )

    # Catch all for models not implemented
    else:
        raise NotImplementedError(
            f"Model '{model_name}' is not implemented. Implemented models: {IMPLEMENTED_MODELS}"
        )

    return model


def load_tokenizer(model_name: str, **kwargs) -> tuple:
    """One-liner for loading a given model's tokenizer from the HuggingFace model hub.

    Args:
        model_name (str): Name of the model to load. Must be one of:
            ['test', 'gpt2', 'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']

    Returns (tuple):
        tokenizer: Tokenizer loaded from the HuggingFace model hub
    """
    if model_name == "test":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", padding_side="left", **kwargs
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Code block for loading small models
    elif model_name in SMALL_MODELS:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", **kwargs
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Code block for loading llama models
    elif "llama" in model_name:
        from transformers import LlamaTokenizer
        from huggingface_hub import hf_hub_download

        tokenizer_location = hf_hub_download(
            repo_id="decapoda-research/llama-7b-hf", filename="tokenizer.model"
        )
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_location, **kwargs)
        tokenizer.pad_token = tokenizer.bos_token

    # Catch all for models not implemented
    else:
        raise NotImplementedError(
            f"Model '{model_name}' is not implemented. Implemented models: {IMPLEMENTED_MODELS}"
        )

    return tokenizer  # type: ignore
