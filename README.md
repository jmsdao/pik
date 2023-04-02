# Evaluating P(IK)

This repository contains the code for experiments aiming to evaluate P(IK) in large language models (probability of "I Know") to assess their calibration (i.e. correctly stating how uncertain they are about their answers).

This builds up on Anthropic's work ['Language Models (Mostly) Know What They Know'](https://arxiv.org/abs/2106.03384), using probing techniques like in ['Discovering Latent Knowledge in Language Models Without Supervision'](https://arxiv.org/abs/2212.03827), where a linear probe $p(IK) = \sigma(v^T h)$ is trained on the hidden activations of the model rather than the model's output.

---

## Environment Setup

Clone the repo and `cd` into it:
```bash
git clone https://github.com/jmsdao/pik.git
cd pik
```

Install the conda environment (use mamba, it's much faster than conda) and activate it:
```bash
mamba env create -f environment.yaml
conda activate pik
```

Install the source packages from this repo:
```bash
pip install -e .
```

Launch your Python interpreter and validate:
```python
python
>>> import pik
>>> pik.ROOT_DIR
PosixPath('/<abs_path_to>/pik')
```

---

## CLI Usage Examples

```bash
# Estimate runtime of an experiment defined by the config
python cli/generate_answers.py --estimate cli/configs/example_generate_answers.yaml
# Run the experiment
python cli/generate_answers.py cli/configs/example_generate_answers.yaml
```

---

## Package Usage Examples

```python
from pik.datasets import load_dataset_and_eval_fn
from pik.models import load_model_and_tokenizer
from pik.models.text_generation import TextGenerator


dataset, eval_fn = load_dataset_and_eval_fn("trivia_qa")
model, tokenizer = load_model_and_tokenizer("gpt2")

tg = TextGenerator(model, tokenizer)
question, answer = dataset[0]

model_answers = tg.generate(question)
eval_fn(model_answers, anwser)  # returns list[int]: 1 if correct, 0 otherwise
```
