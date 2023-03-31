from typing import Optional, Iterable, Union
import torch
from transformers import GenerationConfig


def _get_default_generation_config(tokenizer) -> GenerationConfig:
    """Returns a default generation config for a given tokenizer."""

    if 'gpt2' in str(tokenizer.__class__):
        return GenerationConfig(
            max_new_tokens=16,
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer('\n')['input_ids'][0],
        )

    raise ValueError(f'No default generation config for tokenizer "{tokenizer}"')


class TextGenerator:
    """Generates text using a model."""
    def __init__(self,
        model,
        tokenizer,
        gen_config: Optional[GenerationConfig] = None,
        generation_seed: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.generation_seed = generation_seed

        if gen_config is None:
            self.gen_config = _get_default_generation_config(tokenizer)

    @staticmethod
    def prompt_engineer(prompt_template: str, prompt: str) -> str:
        """Engineers a prompt for a model.

        Args:
            prompt_template (str): text to prepend to the prompt, must
                contain a "{}" to be replaced
            prompt (str): text to use as the prompt

        Returns:
            engineered_prompt (str): text to use as the prompt
        """
        if r'{}' not in prompt_template:
            raise ValueError(r'prompt_template must contain a "{}"')

        return prompt_template.format(prompt)

    def generate(self,
            text_input: str,
            num_generations: int = 1,
            generations_per_pass: int = 1,
        ) -> list[str]:
        """Generate answers (single or multiple) for one question.
        Each model forward pass will be batched by generations_per_pass (at most).
        
        Args:
            text_input (str): text to use as input for the model
            num_generations (int): number of generations total for the given input
            generations_per_pass (int): number of generations to make per pass
                For example, if num_generations=40 and generations_per_pass=15, then
                batch sizes will be [15, 15, 10]

        Returns:
            text_outputs (list[str]): list of generated text with len num_generations.
                Everything before and including the text_input is removed from each output.
        """
        if not isinstance(text_input, str):
            raise ValueError(f'text_input must be a string, not "{type(text_input)}"')
        if num_generations < 1 or generations_per_pass < 1:
            raise ValueError(f'num_generations and generations_per_pass must be >= 1')

        if self.generation_seed:
            torch.manual_seed(self.generation_seed)

        generations_per_pass = min(generations_per_pass, num_generations)

        # Calculate batch sizes used for each pass
        batch_sizes = [generations_per_pass] * (num_generations // generations_per_pass)
        if num_generations % generations_per_pass != 0:
            batch_sizes.append(num_generations % generations_per_pass)

        # Generate text
        text_outputs = []

        for batch_size in batch_sizes:
            batched_text_input = [text_input] * batch_size
            encoded_inputs = self.tokenizer(
                batched_text_input, return_tensors='pt'
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **encoded_inputs,
                    generation_config=self.gen_config,
                )

            text_outputs.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))

        assert len(text_outputs) == num_generations

        # Remove everything before and including the text_input
        start_index = text_outputs[0].index(text_input) + len(text_input)
        text_generations = [text_output[start_index:] for text_output in text_outputs]

        return text_generations

    def generate_multi(self,
            text_inputs: Union[str, Iterable[str]],
            num_generations: int = 1,
        ) -> list[str]:
        """Generate answers (single or multiple) for multiple questions.
        Each model forward pass will be batched by num_generations.

        Args:
            text_inputs (Iterable[str]): list of text to use as input for the model
            num_generations (int): number of generations total for each input

        Returns:
            text_outputs (list[str]): list of generated text. The list has
                len num_generations * num_questions
        """
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        if not isinstance(text_inputs, Iterable):
            raise ValueError(f'text_inputs must be an iterable, not "{type(text_inputs)}"')
        if not all(isinstance(text, str) for text in text_inputs):
            raise ValueError('text_inputs must be an iterable of strings')
        if num_generations < 1:
            raise ValueError(f'num_generations must be >= 1')

        batched_text_inputs = []
        for text in text_inputs:
            batched_text_inputs += [text] * num_generations

        encoded_inputs = self.tokenizer(
            batched_text_inputs, return_tensors='pt'
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded_inputs,
                generation_config=self.gen_config,
            )

        text_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        assert len(text_outputs) == len(batched_text_inputs)

        # Remove everything before and including the text_input
        text_generations = []
        for text_input, text_output in zip(batched_text_inputs, text_outputs):
            start_index = text_output.index(text_input) + len(text_input)
            text_generations.append(text_output[start_index:])

        return text_generations
