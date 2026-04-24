# 01 Environment Fingerprint

## Host

- Windows build: `26200`
- Windows edition: `Windows 10 Pro`
- CPU: `AMD Ryzen 9 7950X`
- Cores / threads: `16 / 32`
- RAM: `126.96 GB`

## GPU

- GPU: `NVIDIA GeForce RTX 4060 Ti`
- VRAM: `16380 MiB`
- Driver: `591.86`

## WSL

- Kernel: `6.6.87.2-microsoft-standard-WSL2`
- Distro: `Ubuntu 24.04.4 LTS`
- Python: `3.12.3`
- `nsys`: `2024.5.1.113`
- `nvcc`: `12.0.140`

## Disk

- `/home/a` filesystem:
  - size: `1007G`
  - used: `137G`
  - available: `819G`

## Suitability conclusion

This machine is suitable for the intended full-version project shape:

- single-GPU TensorRT-LLM runtime work
- real `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- INT4 weight-only engine path
- MoE workload and scheduler evaluations

It is not the target machine for:

- multi-GPU EP
- disaggregated serving
- communication-heavy large-cluster MoE studies
