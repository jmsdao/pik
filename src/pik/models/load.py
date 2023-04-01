import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    LlamaForCausalLM, LlamaTokenizer,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download


SMALL_MODELS = ['gpt2']
LARGE_MODELS = ['llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']
IMPLEMENTED_MODELS = SMALL_MODELS + LARGE_MODELS


def load_model_and_tokenizer(model_name: str) -> tuple:
    """Syntactic sugar for loading a model and tokenizer from the HuggingFace model hub.

    Args:
        model_name (str): Name of the model to load. Must be one of:
            ['test', 'gpt2', 'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']

    Returns (tuple):
        model: Model loaded from the HuggingFace model hub
        tokenizer: Tokenizer loaded from the HuggingFace model hub

    For testing, use model_name='test' to load gpt2 without downloading model
    weights from the HuggingFace model hub (weights will be random).
    """
    if model_name not in IMPLEMENTED_MODELS and model_name != 'test':
        raise NotImplementedError(
            f"Model '{model_name}' is not implemented. Implemented models: {IMPLEMENTED_MODELS}"
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load a small model without downloading files for testing
    if model_name == 'test':
        config = AutoConfig.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_config(config).to(device)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Code block for loading small models
    elif model_name in SMALL_MODELS:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Code block for loading llama models
    elif 'llama' in model_name:
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
        tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)

    return model, tokenizer  # type: ignore
