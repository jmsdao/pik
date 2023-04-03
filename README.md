# Evaluating P(IK)

This repository contains the code for experiments aiming to evaluate P(IK) in large language models (probability of "I Know") to assess their calibration (i.e. correctly stating how uncertain they are about their answers).

This builds up on Anthropic's work ['Language Models (Mostly) Know What They Know'](https://arxiv.org/abs/2106.03384), using probing techniques like in ['Discovering Latent Knowledge in Language Models Without Supervision'](https://arxiv.org/abs/2212.03827), where a linear probe $p(IK) = \sigma(v^T h)$ is trained on the hidden activations of the model rather than the model's output.

---

## Environment Setup

This repo uses mamba, which is a drop-in replacement for conda that is faster and more reliable. You can replace any conda command with mamba and it will work the same. The conda commands are still available with a mambaforge installation.

These instructions assume a fresh install of Ubuntu 20.04. You may need to adjust the commands below depending on your OS and hardware.  

1. Download mambaforge, install it, initialize it, and restart your shell:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
bash Mambaforge-Linux-x86_64.sh -b && \
mambaforge/bin/mamba init bash && \
exec bash
```

2. Clone the repo and `cd` into it:
```bash
git clone https://github.com/jmsdao/pik.git && cd pik
```

3. Edit [`environment.yaml`](https://github.com/jmsdao/pik/blob/main/environment.yaml) to match your hardware (e.g. CUDA version, etc.).  
   > **Note**  
   > If you're using a GPU, uncomment `pytorch-gpu` and select your CUDA version with `pytorch-cuda=<version>`.

4. Create the mamba environment and activate it:
```bash
mamba env create -f environment.yaml && mamba activate pik
```

5. Install the source packages from this repo:
```bash
pip install -e .
```

6. Launch your Python interpreter and check that the `pik` package is installed:
```python
python
>>> import pik
>>> pik.ROOT_DIR
PosixPath('/<abs_path_to>/pik')
```

7. If you're planning to use S3 functionality, make sure you add your AWS credentials to the [`.env`](https://github.com/jmsdao/pik/blob/main/.env) file:

> **Warning**  
> BE CAREFUL NOT TO COMMIT YOUR CREDENTIALS TO GITHUB!

You can run the following command so git doesn't track the `.env` file:
```bash
git update-index --assume-unchanged .env
```

---

## CLI Usage Examples

```bash
# Estimate runtime of an experiment defined by the config
gen-answers --estimate configs/example-gen-answers.yaml
# Run the experiment
gen-answers configs/example-gen-answers.yaml
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

model_answers = tg.generate(question, num_generations=5)
eval_fn(model_answers, anwser)  # returns list[int]: 1 if correct, 0 otherwise
```
