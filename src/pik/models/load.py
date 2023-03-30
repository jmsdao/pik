import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizer,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download


SMALL_MODELS = ['gpt2']
LARGE_MODELS = ['llama-7b']
IMPLEMENTED_MODELS = SMALL_MODELS + LARGE_MODELS


def load_model_and_tokenizer(model_name: str) -> tuple:
    """Loads a model and tokenizer from the Hugging Face model hub.
    
    Args:
        model_name (str): Name of the model to load. Must be one of the following: gpt2, llama-7b

    Returns (tuple):
        model: Model loaded from the Hugging Face model hub
        tokenizer: Tokenizer loaded from the Hugging Face model hub
    """
    if model_name not in IMPLEMENTED_MODELS:
        raise NotImplementedError(
            f"Model '{model_name}' is not implemented. Implemented models: {IMPLEMENTED_MODELS}"
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Code block to load gpt2 model and tokenizer
    if model_name in SMALL_MODELS:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Code block to load llama-7b model and tokenizer
    elif model_name == 'llama-7b':
        checkpoint_location = snapshot_download("decapoda-research/llama-7b-hf")

        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(checkpoint_location)

        model = load_checkpoint_and_dispatch(
            model,  # type: ignore
            checkpoint_location,
            device_map="auto",
            dtype=torch.float16,  # pylint: disable=no-member
            no_split_module_classes=["LlamaDecoderLayer"],
        )
        tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)

    return model, tokenizer  # type: ignore
