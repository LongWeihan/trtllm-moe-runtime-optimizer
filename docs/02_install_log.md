# 02 Install Log

## Goal

Bring up an editable TensorRT-LLM development environment in the full-version workspace.

## Source checkout

- Source root: `/home/a/trtllm-moe-runtime-exp-full/src/TensorRT-LLM`
- Version: `v1.2.1`

The source checkout was created from the already available local TensorRT-LLM clone and then checked out to `v1.2.1`.

## Python environment

- venv: `/home/a/trtllm-moe-runtime-exp-full/venv`

Created with:

```bash
~/.local/bin/virtualenv -p python3 /home/a/trtllm-moe-runtime-exp-full/venv
```

## Editable install

Install command:

```bash
cd /home/a/trtllm-moe-runtime-exp-full/src/TensorRT-LLM
source /home/a/trtllm-moe-runtime-exp-full/venv/bin/activate
TRTLLM_USE_PRECOMPILED=1 pip install -e . --extra-index-url https://pypi.nvidia.com
```

Observed final package state:

- `tensorrt_llm==1.2.1`
- `tensorrt==10.14.1.48.post1`
- `torch==2.9.1`
- `openmpi==5.0.10`

Evidence:

- `/home/a/trtllm-moe-runtime-exp-full/logs/install/install_verify.txt`

## Runtime fixes needed after install

Editable install alone was not sufficient for a stable import path on this machine.

### Fix 1: `libpython3.12.so`

- symlinked `libpython3.12.so` into the venv `lib/` directory

### Fix 2: MPI runtime

- installed user-space `openmpi`

### Fix 3: CUDA 13 runtime loader gap

The editable environment did not initially expose the full set of CUDA 13 runtime libraries needed by TRT-LLM import on this machine. To stabilize the environment, the known-good `nvidia/cu13/lib` contents from the previously validated workspace were synced into the full-version venv.

This is recorded explicitly because it is a real environment fix, not something to hide.

## Final status

The editable TensorRT-LLM development environment is up and usable for the full-version run.
