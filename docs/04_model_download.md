# 04 Model Download

## Fixed main model

- Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- Full-version local path: `/home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat`

## Materialization method

Because the model was already present on the same machine from the validated 24h run, the full-version workspace materialized its own model directory from the existing local source instead of re-downloading from the network.

This keeps the full-version run reproducible on the current machine while avoiding unnecessary duplicate transfer time.

## Result

The model directory is complete and usable for TRT-LLM conversion.

Observed size:

- `27G`

Contains:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- `model-00001-of-00008.safetensors` through `model-00008-of-00008.safetensors`

## Conclusion

The fixed main model is present in the full-version workspace and ready for the official TRT-LLM Qwen path.
