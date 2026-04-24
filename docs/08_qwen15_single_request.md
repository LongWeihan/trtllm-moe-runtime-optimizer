# 08 Qwen15 Single Request

## Goal

Verify that the built full-version engine can execute a real single request.

## Command path

The official TRT-LLM example runner was used:

```bash
python run.py \
  --engine_dir /home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo \
  --tokenizer_dir /home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat \
  --max_output_len 32 \
  --input_text "Explain why MoE routing skew can create decode tail latency."
```

## Result

Single-request inference succeeded.

Evidence:

- [qwen15_single_request.txt](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/results/00_qwen15_sanity/qwen15_single_request.txt)

Observed output excerpt:

`MoE (Multi-Access Edge) routing skew can create decode tail latency ...`

## Conclusion

The fixed full-version model path is stable enough to proceed into benchmark and scheduler work.
