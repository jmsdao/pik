from typing import Optional
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

    def generate(self,
            text_input: str,
            num_generations: int = 1,
            generations_per_pass: int = 1,
        ) -> list[str]:
        """Generate multiple answers for one question.
        
        Args:
            text_input (str): text to use as input for the model
            num_generations (int): number of generations to make total
            generations_per_pass (int): number of generations to make per pass
                For example, if num_generations=40 and generations_per_pass=15, then
                batch sizes will be [15, 15, 10]

        Returns:
            text_outputs (list[str]): list of generated text with len num_generations
        """
        if not isinstance(text_input, str):
            raise ValueError(f'text_input must be a string, not "{type(text_input)}"')

        if self.generation_seed:
            torch.manual_seed(self.generation_seed)

        if generations_per_pass > num_generations:
            generations_per_pass = num_generations

        batch_sizes = [generations_per_pass] * (num_generations // generations_per_pass)
        if num_generations % generations_per_pass != 0:
            batch_sizes.append(num_generations % generations_per_pass)

        text_outputs = []

        for batch_size in batch_sizes:
            batched_text_input = [text_input] * batch_size
            encoded_inputs = self.tokenizer(
                batched_text_input,
                return_tensors='pt'
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **encoded_inputs,
                    generation_config=self.gen_config,
                )

            text_outputs.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))

        start_index = text_outputs[0].index(text_input) + len(text_input)
        text_generations = [text_output[start_index:] for text_output in text_outputs]

        return text_generations

    def generate_multi(self) -> list[str]:
        """Generate multiple answers for multiple questions.
        
        Returns:
            text_outputs (list[str]): list of generated text with len num_generations
        """
        raise NotImplementedError
