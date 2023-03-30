# Evaluating P(IK)

This repository contains the code for experiments aiming to evaluate P(IK) in large language models (probability of "I Know") to assess their calibration (i.e. correctly stating how uncertain they are about their answers).

This builds up on Anthropic's work ['Language Models (Mostly) Know What They Know'](https://arxiv.org/abs/2106.03384), using probing techniques like in ['Discovering Latent Knowledge in Language Models Without Supervision'](https://arxiv.org/abs/2212.03827), where a linear probe $p(IK) = \sigma(v^T h)$ is trained on the hidden activations of the model rather than the model's output.

---

## Environment Setup

Clone the repo and `cd` into it:
```
git clone <repo_url>
cd pik
```

Install the conda environment (use mamba, it's much faster than conda) and activate it:
```
mamba env create -f environment.yml
conda activate pik
```

Install the source packages from this repo:
```
pip install -e .
```

Launch your Python interpreter and validate:
```
python
>>> import pik
>>> pik.ROOT_DIR
PosixPath('/<abs_path_to>/pik')
```

---

## Package Usage Examples

```python
from pik.datasets import trivia_qa

# `dataset` is a subclass of torch.utils.data.Dataset
dataset = trivia_qa.TriviaQADataset()
eval_fn = trivia_qa.evaluate_answer

question, answer_aliases = dataset[0]
model_answer = 'Paris'

eval_fn(model_answer, answer_aliases)  # returns 0 if incorrect, 1 if correct
```

```python
from pik.models import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer('gpt2')
```
