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

3. Edit [`environment.yaml`](https://github.com/jmsdao/pik/blob/main/environment.yaml) to match your hardware (e.g. CUDA version, etc.). You can run `nvidia-smi` to see which CUDA version it's running. Commands are below for convenience:
```bash
nvidia-smi
```

```bash
sudo apt-get install vim  # If you don't have an editor installed
```

   > **Note**  
   > If you're not using any GPUs (why tho??), you can comment out `pytorch-gpu` and `cuda-toolkit=<version>` from the environment YAML to save yourself installation time.




4. Create the mamba environment and activate it:
```bash
mamba env create -f environment.yaml && mamba activate pik
```

5. Check that `which pip` points to the `pik` mamba environment, and correct it if needed:
```bash
which pip  # Should be something like: /home/<user>/mambaforge/envs/pik/bin/pip
```
```bash
export PATH="$CONDA_PREFIX/bin:$PATH"  # If needed
```

6. Install the source packages from this repo:
```bash
pip install -e .
```

7. Launch your Python interpreter and check that the `pik` package is installed:
```python
python
>>> import pik
>>> pik.ROOT_DIR
PosixPath('/<abs_path_to>/pik')
```

8. If you're planning to use S3 functionality, make sure you add your AWS credentials to the [`.env`](https://github.com/jmsdao/pik/blob/main/.env) file:

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
from pik.datasets import load_dataset, get_eval_fn
from pik.models import load_model, load_tokenizer
from pik.models.text_generation import TextGenerator


dataset, eval_fn = load_dataset("trivia_qa"), get_eval_fn("trivia_qa")
model, tokenizer = load_model("gpt2"), load_tokenizer("gpt2")

tg = TextGenerator(model, tokenizer)
question, answer = dataset[0]  # Can also index `dataset` with a slice or iterable

model_answer = tg.generate(question)
eval_fn(model_answer, anwser)  # Returns list[int]: 1 if correct, 0 otherwise
```
