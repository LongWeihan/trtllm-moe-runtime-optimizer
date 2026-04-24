# 07 INT4 WO Build

## Input

- checkpoint dir:
  - `/home/a/trtllm-moe-runtime-exp-full/artifacts/model_conversion/qwen15_moe_int4wo_checkpoint`

## Output

- engine dir:
  - `/home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo`

## Command

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

## Result

Build succeeded.

Important observed build facts:

- total weights memory: `8134053232 bytes`
- engine generation completed in: `16.0983 seconds`
- total build time: `00:00:37`
- build phase peak memory: `18130.80 MB`

Generated files:

- `config.json`
- `rank0.engine`

Observed size:

- engine directory: `7.6G`

Raw log:

- `/home/a/trtllm-moe-runtime-exp-full/logs/build/qwen15_build_int4wo.log`
