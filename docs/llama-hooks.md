The transformers implementation of LLaMA used by this package can be [found here](https://github.com/huggingface/transformers/blob/60d51ef5123d949fd8c59cd4d3254e711541d278/src/transformers/models/llama/modeling_llama.py) (note the commit hash).
Line numbers mentioned herein refer to the above `modelling_llama.py` file.

> **Note**  
> Using llama-7b, I pickled a `dict` that contains every activation (except for `model` and `model.embed_tokens`) for just a single input (`seq_len = 32`, which is approximately the median length for TriviaQA), and the `.pkl` file was ~177MB. We'll have to be quite selective about which activations we want to save (though the main culprit is keeping activations at every token).

---

## LLaMA Sizes

| Size | d_model | nheads | nlayers |
|------|---------|--------|---------|
| 7B   | 4096    | 32     | 32      |
| 13B  | 5120    | 40     | 40      |
| 30B  | 6656    | 52     | 60      |
| 65B  | 8192    | 64     | 80      |

---

## Model Architecture

Code:

```python
from pik.models import load_model, tokenizer

model, tokenizer = load_model("llama-7b"), load_tokenizer("llama-7b")
model
```

Console output:

```bash
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=31999)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

---

## Hookable Modules

Code:

```python
import re
from pik.hooks import HookedModule

hooked_model = HookedModule(model)
modules_to_hook = []

for path in hooked_model.get_hookable_module_paths():
    if not re.findall(r"\.[1-9]", path):  # Omit repeated layers
        modules_to_hook.append(path)

for path in modules_to_hook:
    print(path.replace(".0", ".<layer>"))
```


Console output:

```bash
model
model.embed_tokens
model.layers.<layer>
model.layers.<layer>.self_attn
model.layers.<layer>.self_attn.q_proj
model.layers.<layer>.self_attn.k_proj
model.layers.<layer>.self_attn.v_proj
model.layers.<layer>.self_attn.o_proj
model.layers.<layer>.self_attn.rotary_emb
model.layers.<layer>.mlp
model.layers.<layer>.mlp.gate_proj
model.layers.<layer>.mlp.down_proj
model.layers.<layer>.mlp.up_proj
model.layers.<layer>.mlp.act_fn
model.layers.<layer>.input_layernorm
model.layers.<layer>.post_attention_layernorm
model.norm
lm_head
```

---

## Objects Seen in Forward Hooks

Code:

```python
import torch
from collections import defaultdict
from functools import partial

def fwd_hook_function(module, input, output, module_path=None, store=None) -> None:
    store[f"{module_path}.input"].append(input)
    store[f"{module_path}.output"].append(output)

store = defaultdict(list)
fwd_hook_fns = []
for path in modules_to_hook:
    hook_fn = partial(fwd_hook_function, module_path=path, store=store)
    fwd_hook_fns.append((path, hook_fn))

encoded_input = tokenizer("This has 7 input tokens", return_tensors="pt").to(model.device)
with torch.inference_mode(), hooked_model.hooks(fwd=fwd_hook_fns):
    hooked_model(**encoded_input, output_hidden_states=True, output_attentions=True)

for path in modules_to_hook:
    print(path, "(input):\n")
    stored_obj = store[f"{path}.input"][0]
    hooked_model.print_nested(stored_obj)
    print("\n-----\n")
    print(path, "(output):\n")
    stored_obj = store[f"{path}.output"][0]
    hooked_model.print_nested(stored_obj)
```


Console output (edited for brevity and extra comments):

```bash
model (input):
# Input is input_ids, but the forward hook function doesn't catch it

tuple(
)

-----

model (output):
# Doesn't have anything interesting that isn't caught by other hooks
# Can be handy to aggregate attention weights and final hidden states of all layers

BaseModelOutputWithPast(
    .last_hidden_state:
        torch.Size([1, 7, 4096])
    .past_key_values:  # Omitted if use_cache=False
        tuple(
            32 x tuple(
                torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, d_head)
                torch.Size([1, 32, 7, 128])
            )
        )
    .hidden_states:  # Included if output_hidden_states=True
        tuple(   # .hidden_states[0] is the input embedding, before any decoder layers 
            33 x torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
        )
    .attentions:  # Included if output_attentions=True
        tuple(
            32 x torch.Size([1, 32, 7, 7])  # shape: (batch, nheads, seq_len, seq_len)
        )
)

-----

model.embed_tokens (input):

tuple(
    torch.Size([1, 7])  # shape: (batch, seq_len)
)

-----

model.embed_tokens (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0 (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0 (output):
# The second element of the top level tuple is self_attn_weights (L327).
# Omitted if output_attentions=False in the model's forward function.
# The third element of the top level tuple is cached key values (L330)
# Omitted if use_cache=False.

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
    torch.Size([1, 32, 7, 7])  # shape: (batch, nheads, seq_len, seq_len)
    tuple(
        torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, d_head)
        torch.Size([1, 32, 7, 128])
    )
)

-----

model.layers.0.self_attn (input):
# The input here is the output of input_layernorm (L306-310)
# Not caught by the forward hook function

tuple(
)

-----

model.layers.0.self_attn (output):
# The second element of the top level tuple is attn_weights (L263).
# The second element is None instead if output_attentions=False.
# The third element of the top level tuple is cached key values (L263).
# The third element is None instead if use_cache=False.

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
    torch.Size([1, 32, 7, 7])  # shape: (batch, nheads, seq_len, seq_len)
    tuple(
        torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, d_head)
        torch.Size([1, 32, 7, 128])
    )
)

-----

model.layers.0.self_attn.q_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.self_attn.q_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.self_attn.k_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.self_attn.k_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.self_attn.v_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.self_attn.v_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.self_attn.o_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.self_attn.o_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.self_attn.rotary_emb (input):

tuple(
    torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, d_head)
)

-----

model.layers.0.self_attn.rotary_emb (output):
# cos, sin

tuple(
    torch.Size([1, 1, 7, 128])  # shape: (batch, n_heads_broadcast, seq_len, d_head)
    torch.Size([1, 1, 7, 128])
)

-----

model.layers.0.mlp (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.mlp (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.mlp.gate_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.mlp.gate_proj (output):

torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*d_model)

-----

model.layers.0.mlp.down_proj (input):

tuple(
    torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*d_model)
)

-----

model.layers.0.mlp.down_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.mlp.up_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.mlp.up_proj (output):

torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*d_model)

-----

model.layers.0.mlp.act_fn (input):

tuple(
    torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*d_model)
)

-----

model.layers.0.mlp.act_fn (output):

torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*d_model)

-----

model.layers.0.input_layernorm (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.input_layernorm (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.layers.0.post_attention_layernorm (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.layers.0.post_attention_layernorm (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

model.norm (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

model.norm (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)

-----

lm_head (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, d_model)
)

-----

lm_head (output):
# lm_head is a linear layer (L654), different to embed_tokens
# These are the output logits

torch.Size([1, 7, 32000])  # shape: (batch, seq_len, vocab_size)

-----
```
