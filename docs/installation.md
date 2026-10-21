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

## Disk, GPU memory, and first load

| Resource | Observed behavior |
| --- | --- |
| Official weight download | 19,306,310,880 bytes, plus tokenizer/config files |
| Windows derived shards | Approximately another 19.3 GB; original weights remain intact |
| GPU peak | 7.525 GiB PyTorch allocated in the recorded NF4 smoke, excluding other programs and driver allocations |
| System memory | Loading needs additional system commit headroom; the test host had 32 GB RAM |
| First model call | Loads/quantizes weights; Windows also prepares smaller shards once |
| Later MCP calls | Reuse the model in the server process |

There is no automatic model substitution. Weights and tokenizer are pinned to `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Derived shards change file layout only. The local validation compared dtype, shape, and SHA-256 for all 775 tensors.

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

**Windows error 1455 during loading.** The Windows adapter prepares smaller safetensors shards to avoid the original large-file mapping peak. Leave space for both caches and enough available system memory. The conversion does not alter system paging settings. Interrupted conversion is rebuilt under a lock before its completed index is published.

**The first request takes time.** It includes load/quantization. Give the MCP client a generous tool timeout (the integration script uses five minutes). Use a persistent MCP process for a writing session.

**Missing optional attention kernels.** Windows validation used the reference PyTorch implementations. Warnings about `causal_conv1d` or `flash-linear-attention` did not prevent the recorded smoke from passing; no optimized-kernel performance is claimed.

**Unexpected lyric or weak rhyme.** Generation samples a finite set. Inspect pronunciation explanations, add a sung pronunciation override, supply better alternatives to `rank_candidates`, or adjust `strength`. Unknown words remain with an explicit weight; they are not automatically pronounced.

**Input too long.** The adapter caps model input at 2,048 tokens and raises an error instead of silently truncating. Keep enough room for the prompt, gap, and surrounding text.

## Reproduce checks

See [CONTRIBUTING.md](../CONTRIBUTING.md) for core tests and the opt-in real-model scripts. [verification.md](verification.md) records what actually ran; CI does not download the 9B weights.
