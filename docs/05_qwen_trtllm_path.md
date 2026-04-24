# 05 Qwen TRT-LLM Path

## Source of truth

The official local reference used for this full-version run is:

- `/home/a/trtllm-moe-runtime-exp-full/src/TensorRT-LLM/examples/models/core/qwen/README.md`

## Confirmed support

The support matrix in the Qwen README lists:

- `Qwen1.5-MoE-A2.7B(-Chat)` as supported
- `WO` (weight-only) as supported
- architecture requirement: `Ampere+`

This matches the project constraints:

- real Qwen MoE model
- TRT-LLM-supported path
- feasible on `RTX 4060 Ti`

## Relevant entry points

- conversion script:
  - `examples/models/core/qwen/convert_checkpoint.py`
- inference helper:
  - `examples/run.py`
- engine build command:
  - `trtllm-build`

## Chosen full-version commands

### Conversion

```bash
python convert_checkpoint.py \
  --model_dir /home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat \
  --output_dir /home/a/trtllm-moe-runtime-exp-full/artifacts/model_conversion/qwen15_moe_int4wo_checkpoint \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4
```

### Build

```bash
trtllm-build \
  --checkpoint_dir /home/a/trtllm-moe-runtime-exp-full/artifacts/model_conversion/qwen15_moe_int4wo_checkpoint \
  --output_dir /home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo \
  --gemm_plugin float16 \
  --max_batch_size 4 \
  --max_input_len 128 \
  --max_seq_len 256 \
  --max_num_tokens 512
```

## Why this path was kept

This is the most faithful path to the project story:

- fixed real Qwen MoE model
- official TRT-LLM conversion/build flow
- INT4 weight-only as the formal quantized execution route
