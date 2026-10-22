# Verification record

[Home](../README.md) · [Reproduce the checks](../CONTRIBUTING.md)

Initial 9B GPU measurements were recorded on **2026-10-20**; 4B measurements and updated core checks were recorded on **2026-10-21**. These are functional integration records, not comparisons with other writing tools or a listening evaluation.

## Core behavior

- **73 core pytest tests pass** on Windows / Python 3.10, including a fresh environment without torch.
- A real MCP stdio subprocess initializes, discovers six tools, compares pronunciations, handles custom rules, retains unknown candidates, returns phonetic-only rankings and model state, and reports invalid inputs.
- Tests also check 4B/9B model selection, exact snapshot pins, matching tokenizer/model load arguments, and real stdio model status for both selections. Tests cover positive soft weights, pair direction and precedence, positional rules, no transitive closure, alternative readings, multiple syllables, unknown retention, syllable preferences, and strength zero.
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

## 2026-10-21 follow-up resource check (resolved in 0.2.1)

During 4B adaptation, the current host failed to reload 9B with a MemoryError while materializing its first 2.03 GB embedding. System commit headroom before model load was about 11.74 GB and fell to 3.48 GB before that read; the safetensors pread slice path requires additional temporary copies. Running the unmodified initial commit on the same host reproduced the identical failure. The dated successful 9B records above remain intact; they are not a claim that the later resource-limited run succeeded.

## Actual Qwen3.5-4B integration — 2026-10-21

| Setting | Recorded value |
| --- | --- |
| Model | Qwen/Qwen3.5-4B |
| Weights/tokenizer revision | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| OS / Python / GPU | Windows / 3.10.11 / RTX 5080 |
| Torch / quantization | 2.8.0+cu128 / NF4 with double quantization, BF16 compute |
| Tokenizer entries | 248077, loaded from the selected 4B snapshot |
| Original weight bytes | 9,319,828,096 in two SHA-256-verified shards |
| Peak PyTorch allocated GPU memory | 3.217 GiB |
| Final direct smoke, existing derived cache | 12.56 seconds |
| Final production-entrypoint MCP smoke | 14.68 seconds |

Raw outputs: [model-smoke-4b.json](model-smoke-4b.json), [mcp-model-smoke-4b.json](mcp-model-smoke-4b.json), [reshard-check-4b.json](reshard-check-4b.json).

Both integration scripts actually loaded the selected official 4B weights and scored/generated text. The direct check also ranks multi-token gap fills with right context. An independent streaming comparison of all **738 tensors** found identical dtype, shape, and per-tensor SHA-256 between source and derived files. The existing 9B raw records were not overwritten.

Initial 4B suggestions were valid JSON but too generic for the Chinese brief about insomnia and being afraid to call. The shared prompt now explicitly preserves concrete meaning, treats the rhyme reference as a sound rather than a topic, and reinforces the JSON-array output contract. The direct smoke, with seed 42, first failed a narrow lexical assertion for call/phone/dial/number/ring, then passed after the prompt adjustment. The final MCP exchange passed the same brief check. This lexical assertion is specific to the test brief and is not a general meaning or lyric-quality metric.

Final direct examples include `can't make a call tonight` and `dread the phone ring tonight`; the MCP output includes `too scared to call on this long night`. Other candidates still generalize the mood, so editorial selection remains necessary. Sampling can produce different text or malformed output on other requests; the adapter reports unparseable proposals explicitly.

The recorded 4B times include model loading, scoring, and generation with the derived cache prepared. They are functional measurements on one host, not a controlled 4B/9B quality or speed benchmark, nor a guarantee that every 4 GB GPU can run the model.

## Memory fix verification — 0.2.1

The later 9B loading failure is now resolved on the measured host. It involved avoidable dependency staging as well as memory headroom. Failed-load traceback retention and a workspace-pinned reserved CUDA block were also reproduced and fixed. Both sizes passed injected failure recovery and two same-process load/score/unload cycles, finishing each explicit `release_runtime_cache=True` unload at 1 KiB allocated and 2 MiB reserved. That option requires external CUDA work to be stopped; default cleanup does not purge shared cuBLAS workspaces. See [the full memory audit](model-memory.md) and its raw records.

Updated validation: 47 core tests in a torch-free environment (one optional module skipped); 57 tests including 10 real-tensor streaming tests in the model environment. The latter cover BF16/F16/F32 bytes, matrices, scalar/empty tensors, and truncated input. Functional direct and production MCP runs passed for both 9B and 4B after the fix. Existing historical JSON remains intact.

## Suggest-first and inference-error recovery — 0.2.2

The live failed Cherry Studio 9B server retained weights after the reported cublasCreate allocation error. A fresh server accepted the same request before the fix, so the native driver failure was not deterministic. Confirmed gaps were deferred runtime initialization, missing suggest-first coverage, and no cleanup on inference allocation errors.

The fix prepares cuBLAS/BF16 before NF4 weights, returns unused loading reservations, and clears allocation-owning inference frames/weights on allocation failure. Real 9B/4B production MCP processes now have records for the city-lights request as their first model operation, a second request in the same process, and a separately labeled injected-error recovery audit. See [suggest-reliability.md](suggest-reliability.md) and its four raw records. Tests now total 50 core plus 10 optional real-tensor tests.

## Token-path search and balanced lookup — 0.3.0

The sixth MCP tool `search_lyrics` searches actual next-token paths, with default 20 active paths / top20 extensions / 12 new-token budget. It returns original sequence probability separately from custom soft rhyme weights; it does not request a subjective semantic score from Qwen. Full algorithm, budget, EOS, subword and short-path semantics are in [probability-search.md](probability-search.md).

| Actual model (Windows / RTX 5080 / NF4) | Direct full-beam including load | Peak allocated GPU | Cold production MCP including load |
| --- | --- | --- | --- |
| Qwen3.5-4B | 22.15 s | 3.157 GiB | 24.24 s |
| Qwen3.5-9B | 31.17 s | 7.473 GiB | 32.46 s |

Each direct check independently teacher-forced the first three returned token paths. Maximum absolute logprob error was below 0.000024 for 4B and 0.0000025 for 9B. Probabilities are normalized against the full vocabulary before nontext filtering and top-k selection. Both models evaluated the full default search, not a reduced-width test.

Batch-size changes can alter BF16/NF4 rounding: the recorded 9B logprob difference for one top token between microbatch 1 and 2 is approximately 0.123. The check records per-token differences and uses a stated 0.15 logprob tolerance; it does not promise bitwise equality or unchanged beam outcomes across batching. Default microbatch is 1 and does not reduce the 20-path frontier.

Raw records: [4B direct](search-model-4b.json), [9B direct](search-model-9b.json), [4B MCP](search-mcp-4b.json), [9B MCP](search-mcp-9b.json). The cold MCP check calls search before any other model operation. These are functional/probability tests, not a comparative quality benchmark; pure probability can prefer short phrases and finite pruning can miss rhyme families.

Balanced dictionary lookup tests cover mine/fall useful common terms, custom rule preservation, raw phonetic mode and expandable same-pronunciation variants. They use installed wordfreq English data locally; its code and data licenses are acknowledged separately. No online search was used in this change.

Final 0.3.0 checks: 73 tests pass in the torch-free core environment (one optional module skipped); 83 tests pass in the model environment including real tensor-transfer tests. Ruff, dependency compatibility, documentation links, example JSON, sdist/wheel builds and package contents passed. Both actual-model and MCP search records report package_version=0.3.0.

## Windows allocator warmup fix — 0.3.1 (2026-10-22)

The exact 7,752,651,232-byte request was traced to the pinned Transformers allocator warmup before 9B weights load. A native 7.22 GiB request failed with 14.64 GiB GPU memory reported free. Windows GraphicsPerfSvc separately retained roughly 54.4 GiB of committed memory; after the user-approved service restart and natural old-host exit, native 7.22 GiB allocation succeeded on the same GPU/driver. A universal fixed 4 GiB WDDM ceiling was not established.

The Windows loader now caps the optional disposable warmup at 256 MiB through a thread-local dispatch mode, preserving every actual tensor shape and the 20×20 search. Unit tests check ordinary allocations/small warmups are unchanged, thread isolation and exception restoration. Core tests: 73 passed, 2 optional modules skipped. Model environment: 86 passed (10 tensor-transfer tests and 3 allocator/fallback-bound tests included).

Actual 9B and 4B full default searches passed with a 4 GiB single-factory allocation ceiling enforced by an external audit. Actual fallback at maximum supported search shape B8/L2048/H32 produced finite outputs with a 128 MiB pairwise masked_fill, maximum observed tensor 256 MiB, and isolated-layer peak 3.298 GiB. No attention algorithm was replaced. Independent 9B path-probability checks and cold production MCP search passed after the loader fix. Detailed scope, raw JSON, resource limits and reproduction commands: [windows-allocation.md](windows-allocation.md).
