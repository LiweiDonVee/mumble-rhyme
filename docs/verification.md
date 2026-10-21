# Verification record

[Home](../README.md) · [Reproduce the checks](../CONTRIBUTING.md)

GPU measurements were recorded on **2026-10-20**. Public-release core checks were repeated on **2026-10-21**. These are functional integration records, not comparisons with other writing tools or a listening evaluation.

## Core behavior

- **35 pytest tests pass** on Windows / Python 3.10, including a fresh environment without torch.
- A real MCP stdio subprocess initializes, discovers five tools, compares pronunciations, handles custom rules, retains unknown candidates, returns phonetic-only rankings and model state, and reports invalid inputs.
- Tests cover positive soft weights, pair direction and precedence, positional rules, no transitive closure, alternative readings, multiple syllables, unknown retention, syllable preferences, and strength zero.
- A review found and fixed a syllable count that previously came from a different pronunciation than the selected match. Another regression covers an interrupted repair of a derived checkpoint cache.
- Ruff and sdist/wheel builds pass. Public documentation links and example JSON have been checked locally.
- The real pinned tokenizer has 248077 entries. Full-text offset selection includes boundary-crossing tokens and right context.
- A separate adapter check uses this tokenizer and a random tiny Qwen3.5 with linear/full attention. Adapter scores agree with independent full-logits scores, and cached generation runs. This tests API compatibility separately from real 9B inference.

## Actual Qwen3.5-9B integration

| Setting | Recorded value |
| --- | --- |
| Model | Qwen/Qwen3.5-9B |
| Weights/tokenizer revision | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| OS / Python | Windows / 3.10.11 |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| Torch | 2.8.0+cu128 |
| Quantization | bitsandbytes NF4, double quantization, BF16 compute |
| Peak PyTorch allocated GPU memory | 7.525 GiB |
| Direct smoke, initial derived-cache preparation included | 55.18 seconds |
| Direct smoke, derived cache already prepared | 27.71 seconds |
| Real MCP smoke through the production serve entrypoint | 26.35 seconds |

The requests differ between the direct and MCP scripts. These times include loading, ranking, and generation; they are not generation-only latency or token-throughput measurements. PyTorch allocated memory excludes other applications and driver allocations. The test host had 32 GB of system RAM.

Raw results: [model-smoke.json](model-smoke.json), [mcp-model-smoke.json](mcp-model-smoke.json).

The direct smoke ranks `head` above `chest` and `dreams` in `I keep your voice inside my ___`, with `bed` as the reference. It then generates English lines from a Chinese creative brief and ranks them. Recorded candidates include `Talking till the early morning light` and `Daring to disturb your sleep tonight`; both have a final-phonetic weight of 1 against `night`. These examples demonstrate the pipeline, not editorial quality.

The MCP smoke calls custom-profile comparison, actual model ranking, actual model suggestions, and model_status. It confirms loaded=true and preserves structured results through a real stdio client/server exchange.

## Download and Windows loading

- Four original shards total **19,306,310,880 bytes**, each verified against its upstream SHA-256.
- Windows initially failed with error 1455 when safetensors temporarily mapped a large shard during load. Smaller standard safetensors shards avoid that observed peak; source files remain intact.
- An independent streaming comparison checked **all 775 tensors** across original and derived layouts: dtype, shape and per-tensor SHA-256 match exactly.
- The cache uses a lock, invalidates incomplete-cache completion state before repair, and publishes its final index after successful writes.
- Windows also stalled while first importing NumPy after stdio threads were started. The serve entrypoint now preimports optional NumPy before the threads, while leaving torch and weights lazy. The production entrypoint was then verified by the actual model MCP smoke.
- No system paging settings were modified. The model environment is isolated from globally installed torch.

## Interpretation and scope

English pronunciation comes from dictionary readings and hand-designed similarity, with user overrides. Chinese is accepted as a creative brief; Chinese phonetic scoring and recording analysis are absent. Bounded proposals can miss useful unconventional rhymes; custom pair rules apply at reranking time.

Windows uses reference PyTorch kernels where causal_conv1d and flash-linear-attention are absent. The complete 9B smoke passed with those fallbacks. No optimized-kernel performance, universal hardware minimum, or subjective songwriting superiority is claimed.
