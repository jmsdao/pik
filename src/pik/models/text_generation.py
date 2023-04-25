from typing import Optional, Iterable, Union, Generator
import torch
from transformers import GenerationConfig
from pik.datasets.utils import chunked


def _get_default_generation_config(tokenizer) -> GenerationConfig:
    """Returns a default generation config for a given tokenizer."""

    if "gpt2" in str(tokenizer.__class__).lower():
        return GenerationConfig(
            max_new_tokens=16,
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer("\n")["input_ids"][0],
        )

    elif "llama" in str(tokenizer.__class__).lower():
        return GenerationConfig(
            max_new_tokens=16,
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=13,  # llama's newline token id
        )

    raise ValueError(f'No default generation config for tokenizer "{tokenizer}"')


class TextGenerator:
    """Handy wrapper for text generation."""

    def __init__(
        self,
        model,
        tokenizer,
        gen_config: Optional[Union[dict, GenerationConfig]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config

        if gen_config is None:
            self.gen_config = _get_default_generation_config(tokenizer)
        if isinstance(gen_config, dict):
            self.gen_config = GenerationConfig(**gen_config)

        self.model.eval()

    @staticmethod
    def prompt_engineer(
        prompt_template: str, prompt: Union[str, list[str]]
    ) -> Union[str, list[str]]:
        """Engineers a prompt for a model.

        Args:
            prompt_template (str): template for the prompt. eg. 'Q: {} A:'
                Must include a "{}" to be replaced with the prompt
            prompt (str or list[str]): text to use as the prompt

        Returns:
            engineered_prompt (str): text to use as the prompt
        """
        if r"{}" not in prompt_template:
            raise ValueError(r'prompt_template must contain a "{}"')

        if isinstance(prompt, list):
            return [prompt_template.format(p) for p in prompt]

        return prompt_template.format(prompt)

    @staticmethod
    def get_batched_data_ids(
        data_ids: Iterable[int],
        num_generations: int,
        batch_size: int,
    ) -> Generator:
        """Returns a list of batched data ids.

        Args:
            data_ids (Iterable[int]): list of data ids
            num_generations (int): number of generations per data id
            batch_size (int): max batch size per model forward pass

        Returns:
            batched_ids (Generator): generator of batched data ids

        Example:
        ```
        >>> data_ids = [0, 1, 2, 3]
        >>> batched_ids = TextGenerator.get_batched_data_ids(data_ids, 3, 5)
        >>> [batch for batch in batched_ids]
        [[0, 0, 0, 1, 1], [1, 2, 2, 2, 3], [3, 3]]
        ```
        """
        if num_generations < 1 or batch_size < 1:
            raise ValueError("num_generations and batch_size must be >= 1")

        repeated_ids = [i for i in data_ids for _ in range(num_generations)]
        batched_ids = chunked(repeated_ids, batch_size)
        return batched_ids

    def generate(
        self,
        text_inputs: Union[str, list[str]],
        remove_input: bool = True,
    ) -> list[str]:
        """Generate answers for a batch of questions.

        Args:
            text_inputs (str or list[str]): text to use as input for the model
            remove_input (bool): whether to remove the input text from the output

        Returns:
            text_outputs (list[str]): list of generated text with input text removed
        """
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        encoded_inputs = self.tokenizer(
            text_inputs, return_tensors="pt", padding=True
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded_inputs,
                generation_config=self.gen_config,
            )

        text_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove the input strings and everything before from the output strings
        if remove_input:
            # Pass the input through the tokenizer in case the input text length changes
            text_inputs_decoded = self.tokenizer.batch_decode(
                encoded_inputs["input_ids"], skip_special_tokens=True
            )
            for i, text_input in enumerate(text_inputs_decoded):
                text_outputs[i] = text_outputs[i][len(text_input):]

        # Unlist if only one input
        if len(text_inputs) == 1:
            text_outputs = text_outputs[0]

        return text_outputs
