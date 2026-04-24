# 06 Conversion Log

## Input

- HF model path:
  - `/home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat`

## Output

- TRT-LLM checkpoint path:
  - `/home/a/trtllm-moe-runtime-exp-full/artifacts/model_conversion/qwen15_moe_int4wo_checkpoint`

## Command

```bash
python /home/a/trtllm-moe-runtime-exp-full/src/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat \
  --output_dir /home/a/trtllm-moe-runtime-exp-full/artifacts/model_conversion/qwen15_moe_int4wo_checkpoint \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4
```

## Result

Conversion succeeded.

Observed log summary:

- total conversion time: `00:03:23`

Generated files:

- `config.json`
- `rank0.safetensors`

Observed size:

- checkpoint directory: `7.6G`

Raw log:

- `/home/a/trtllm-moe-runtime-exp-full/logs/build/qwen15_convert_int4wo.log`
