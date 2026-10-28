# Installation and troubleshooting

[Home](../README.md) · [中文](../README.zh-CN.md) · [MCP & API](api.md)

## Core: no model, no GPU

Use Python 3.10 or newer. Clone/download this repository, open a terminal in its root, and create an isolated environment:

```console
git clone https://github.com/LiweiDonVee/mumble-rhyme.git
cd mumble-rhyme
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv/Scripts/Activate.ps1
```

Linux/macOS:

```sh
source .venv/bin/activate
```

If activation is unavailable, run `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on POSIX in place of `python`, followed by `-m mumble_rhyme` for CLI commands.

```console
python -m pip install -e .
python -m mumble_rhyme --profile examples/mumble.yaml compare sit set
python -m mumble_rhyme find night --limit 12
python -m mumble_rhyme status
```

These paths use the installed pronunciation dictionary. They do not download Qwen. To rank supplied phrases without the model, use a rank JSON request with `"use_model": false`, or pass it to the MCP tool.

## Model: tested Windows CUDA setup

The actual GPU validation used Windows, Python 3.10.11, RTX 5080 (16 GB), torch 2.8.0+cu128, transformers 5.20.2, bitsandbytes 0.50.2. With the environment active:

```console
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[model]"
python -m mumble_rhyme download
python -m mumble_rhyme --profile examples/mumble.yaml suggest --request examples/suggest.json
```

Install CUDA torch first. An older CUDA build can detect the RTX 5080 yet fail on actual operations. `requirements-lock.txt` records the verified Windows/Python 3.10 environment; it is a version snapshot, not a portable hash lock. It is optional for installation.

Other NVIDIA platforms need a compatible PyTorch build and bitsandbytes support. The complete 9B path has been verified on the Windows configuration above; Linux/macOS GPU inference has not been measured here. Core code is portable Python; CI exercises the core separately. `--quantization none` disables NF4 and uses automatic device placement, with much larger memory needs; CPU-only 9B generation is unverified.

## Choose 9B or 4B

`--model` is a global option and must come before the subcommand. The default is `9b`, preserving existing commands. Use `4b` explicitly:

```console
python -m mumble_rhyme --model 4b download
python -m mumble_rhyme --model 4b status
python -m mumble_rhyme --model 4b --profile examples/mumble.yaml suggest --request examples/suggest.json
python -m mumble_rhyme --model 4b --profile examples/mumble.yaml serve
```

| Selection | Fixed revision for both weights and tokenizer | Original weights |
| --- | --- | --- |
| `9b` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 19,306,310,880 bytes |
| `4b` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9,319,828,096 bytes |

Models have separate snapshots and derived caches. Selecting 4B does not download or replace 9B. Each MCP server keeps its startup selection; tools report that model ID and revision. Example 4B MCP configurations: [Windows](../examples/mcp.windows.4b.json), [POSIX](../examples/mcp.posix.4b.json).

## Disk, GPU memory, and first load

| Resource | Observed behavior |
| --- | --- |
| Official weight download | 19.31 GB for 9B, 9.32 GB for 4B, plus tokenizer/config files |
| Windows derived shards | Approximately another copy of the selected weights; original weights remain intact |
| Recorded 4B GPU peak | 3.217 GiB PyTorch allocated in the final 4B smoke; complete cached direct smoke 12.56 s, MCP 14.68 s |
| Recorded 9B GPU peak | 7.525 GiB PyTorch allocated in the dated NF4 smoke, excluding other programs and driver allocations |
| System memory | Loading needs additional system commit headroom; the test host had 32 GB RAM |
| First model call | Loads/quantizes weights; Windows also prepares smaller shards once |
| Later MCP calls | Reuse the model in the server process |

There is no automatic model substitution. Each selection pins weights and tokenizer to the same revision listed above. Derived shards change file layout only. The original 9B validation compared dtype, shape, and SHA-256 for all 775 tensors. Actual model records are kept separately in [verification.md](verification.md).

## Cache and offline use

Hugging Face controls the model cache. Set `HF_HOME` **before** downloading if you want another disk. On Windows the derived cache is inside the same model cache tree, under `mumble-rhyme-shards/<revision>`.

Once the snapshot is complete, force offline mode if desired:

```powershell
$env:HF_HUB_OFFLINE = "1"
python -m mumble_rhyme --profile examples/mumble.yaml serve
```

POSIX: `HF_HUB_OFFLINE=1 python -m mumble_rhyme serve`.

Inference and phonetic scoring run locally. Normal online model loading can contact Hugging Face for files/metadata. If a cloud assistant calls your MCP server, that assistant may receive the text and tool results you give it. The project does not add an analytics or lyric-upload service.

## Common problems

**“NF4 requires CUDA.”** Core commands still work. Install compatible CUDA torch for the model path, or explicitly choose `--quantization none` before the subcommand with sufficient memory.

**Windows error 1455 during loading.** The Windows adapter prepares smaller safetensors shards to avoid the original large-file mapping peak. Leave space for both caches and enough available system memory. The conversion does not alter system paging settings. Version 0.2.1 also avoids the full-tensor duplication and bounds large NF4 tensor staging to a 16 MiB CPU buffer. It fixes the later 9B reload failure and provides explicit full-cache unload for workspace-pinned unused CUDA reservations (with other same-process CUDA work stopped). Both models passed same-process retry and repeated unload checks; see the [memory audit](model-memory.md). Adequate system RAM/commit and GPU memory are still required. Interrupted conversion is rebuilt under a lock before its completed index is published.

**The first request takes time.** It includes load/quantization. Give the MCP client a generous tool timeout (the integration script uses five minutes). Use a persistent MCP process for a writing session.

**`CUBLAS_STATUS_ALLOC_FAILED` on first suggestion.** Version 0.2.2 prepares cuBLAS/BF16 before loading weights and cleans up weights after inference allocation failure. Restart the server after updating and check `model_status.package_version`. Avoid duplicate 9B servers on the same GPU. A later request can reload once resources are available; choose `--model 4b` explicitly if desired. See the [investigation and its limits](suggest-reliability.md).

**Missing optional attention kernels.** Windows validation used the reference PyTorch implementations. Warnings about `causal_conv1d` or `flash-linear-attention` did not prevent the recorded smoke from passing; no optimized-kernel performance is claimed.

**Unexpected lyric or weak rhyme.** Generation samples a finite set. Inspect pronunciation explanations, add a sung pronunciation override, supply better alternatives to `rank_candidates`, or adjust `strength`. Unknown words remain with an explicit weight; they are not automatically pronounced.

**Input too long.** The adapter caps model input at 2,048 tokens and raises an error instead of silently truncating. Keep enough room for the prompt, gap, and surrounding text.

## Reproduce checks

See [CONTRIBUTING.md](../CONTRIBUTING.md) for core tests and the opt-in real-model scripts. [verification.md](verification.md) records what actually ran; CI does not download model weights. All three model validation scripts accept `--model 4b` or `--model 9b`.

## Version 0.3 search and lookup

Core installation now includes wordfreq and its local data for balanced lookup. No online word-frequency service is used. The sixth tool search_lyrics uses the selected Qwen model; its beam width defaults to 20 independently of microbatch size 1. Test scripts use the full default 20×20 budget. See [probability-search.md](probability-search.md) for raw probability, boundary and length-limit behavior. After updating, restart the MCP process and check that `model_status.package_version` matches the installed package (currently 0.3.1) and that six tools are registered.

## Large CUDA request despite free GPU capacity

Version 0.3.1 limits only the Windows Transformers disposable allocator warmup to 256 MiB; 9B previously requested 7.22 GiB in a single speed-only allocation before reading weights. This does not reduce beam width or real parameter shapes. Check model_status.loading_warmup after restarting the server.

Do not infer a fixed WDDM allocation ceiling from one failure. Local native 7.22 GiB allocation failed under substantial system commit pressure and succeeded after a user-approved GraphicsPerfSvc restart on the same GPU/driver. The package does not change system services or paging. See [windows-allocation.md](windows-allocation.md) for trace, probes, supported fallback bounds and remaining memory requirements.
