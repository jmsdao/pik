## Steps for adding a new model
1. In `pik/models/load.py`:
    - Add the model name to the `LARGE_MODELS` if you need to use something like accelerate to load the model. Otherwise, add it to the `SMALL_MODELS` list.
    - Add an `if` block to load the model in the `load_model` function
    - Add an `if` block to load it's tokenizer in the `load_tokenizer` function. Tokenizer loading should work even when the full model has not been downloaded yet.
2. In `pik/models/text_generation.py`, add a default generation config to the function `_get_default_generation_config`
