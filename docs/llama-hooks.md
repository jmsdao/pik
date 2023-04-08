The transformers implementation of LLaMA used by this package can be [found here](https://github.com/huggingface/transformers/blob/60d51ef5123d949fd8c59cd4d3254e711541d278/src/transformers/models/llama/modeling_llama.py) (note the commit hash).
Line numbers mentioned herein refer to the above `modelling_llama.py` file.

> **Note**  
> Using llama-7b, I pickled a `dict` that contains every activation (except for `model` and `model.embed_tokens`) for just a single input (`seq_len = 32`, which is approximately the median length for TriviaQA), and the `.pkl` file was ~177MB. We'll have to be quite selective about which activations we want to save (though the main culprit is keeping activations at every token).

---

Hookable modules (`layer` is an integer from 0 to 31 inclusive):
```bash
model
model.embed_tokens
model.layers  # Doesn't activate hooks (maybe because it's a nn.ModuleList?)
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

Example (from llama-7b) of input/output objects seen in the forward hook function:
```bash
model (input):
# Input is input_ids, but the forward hook function doesn't catch it

tuple(
)

-----

model (output):
# Doesn't have anything interesting that isn't caught by other hooks
# Can be handy to aggregate attention weights and final hidden states of
# all layers

BaseModelOutputWithPast

-----

model.embed_tokens (input):

tuple(
    torch.Size([1, 7])  # shape: (batch, seq_len)
)

-----

model.embed_tokens (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0 (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0 (output):
# The second element of the top level tuple is self_attn_weights (L327).
# Omitted if output_attentions=False in the model's forward function.
# The third element of the top level tuple is cached key values (L330)
# Omitted if use_cache=False.

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
    torch.Size([1, 32, 7, 7])  # shape: (batch, nheads, seq_len, seq_len)
    tuple(
        torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, head_dims)
        torch.Size([1, 32, 7, 128])
    )
)

-----

model.layers.0.self_attn (input):
# The input is the output of input_layernorm (L306-310)
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
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
    torch.Size([1, 32, 7, 7])  # shape: (batch, nheads, seq_len, seq_len)
    tuple(
        torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, head_dims)
        torch.Size([1, 32, 7, 128])
    )
)

-----

model.layers.0.self_attn.q_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.self_attn.q_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.self_attn.k_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.self_attn.k_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.self_attn.v_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.self_attn.v_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.self_attn.o_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.self_attn.o_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.self_attn.rotary_emb (input):

tuple(
    torch.Size([1, 32, 7, 128])  # shape: (batch, nheads, seq_len, head_dims)
)

-----

model.layers.0.self_attn.rotary_emb (output):
# cos, sin

tuple(
    torch.Size([1, 1, 7, 128])  # shape: (batch, n_heads_broadcast, seq_len, head_dims)
    torch.Size([1, 1, 7, 128])
)

-----

model.layers.0.mlp (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.mlp (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.mlp.gate_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.mlp.gate_proj (output):

torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*hidden_dims)

-----

model.layers.0.mlp.down_proj (input):

tuple(
    torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*hidden_dims)
)

-----

model.layers.0.mlp.down_proj (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.mlp.up_proj (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.mlp.up_proj (output):

torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*hidden_dims)

-----

model.layers.0.mlp.act_fn (input):

tuple(
    torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*hidden_dims)
)

-----

model.layers.0.mlp.act_fn (output):

torch.Size([1, 7, 11008])  # shape: (batch, seq_len, 4*hidden_dims)

-----

model.layers.0.input_layernorm (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.input_layernorm (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.layers.0.post_attention_layernorm (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.layers.0.post_attention_layernorm (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

model.norm (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

model.norm (output):

torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)

-----

lm_head (input):

tuple(
    torch.Size([1, 7, 4096])  # shape: (batch, seq_len, hidden_dims)
)

-----

lm_head (output):
# lm_head is a linear layer (L654), different to embed_tokens

torch.Size([1, 7, 32000])  # shape: (batch, seq_len, vocab_size)

-----
```
