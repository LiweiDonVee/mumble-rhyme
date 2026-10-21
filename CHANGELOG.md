# Changelog

## 0.3.0

- Add search_lyrics / CLI search: bounded token-tree search with default 20 active paths and top-20 extensions, raw sequence probabilities, completed-word soft rhyme guidance, explicit length normalization and cutoff metadata.
- Keep microbatch size independent from search width; record actual 4B/9B probabilities against teacher-forced checks and cold MCP calls. No model self-grading.
- Improve find_rhymes with default local word-frequency balancing and expandable same-pronunciation alternatives; retain raw phonetic mode.
- Add wordfreq code/data attribution and document finite search, short-path bias, incomplete word limits, and quantized batch-size numeric drift.

## 0.2.2

- Initialize cuBLAS and a real BF16 linear operation before NF4 weights occupy memory; release unused loading reservations before first inference.
- Clean up model weights and exception-held tensors on generation/scoring allocation failures, not just failed loading. Preserve the reported error and return an explicit recovery message; no automatic model substitution or hidden retry.
- Include the running package version and last resource error in model status.
- Add the reported city-lights request as a cold suggest-first MCP check, plus real-model fault injection followed by same-process retry for both model sizes.

## 0.2.1

- Reduce Windows NF4 load peaks: validated large floating-point tensors stream directly to CUDA using a 16 MiB CPU buffer; other tensors use single-buffer full-tensor reads.
- Release partially loaded models retained by exception cause/context chains, close readers on all load exits, and clear unused CUDA cache after failure.
- Add explicit Python `LocalModel.unload()` and serialize project loading/inference/unloading with a shared lock. Optional full runtime-cache release requires other same-process CUDA work to be stopped.
- Add real failure/retry/load/unload memory audits, bit-preserving streaming checks, and fresh 4B/9B generation/MCP records.

## 0.2.0

- Add explicit Qwen3.5-4B selection through `--model 4b` and `LocalModel(model="4b")`; 9B remains the default.
- Pin each model and its matching tokenizer independently; keep model and derived caches separate.
- Expose the selected model through MCP instructions and status; provide 4B MCP configuration examples.
- Parameterize real-model, tokenizer-adapter, and MCP smoke scripts, preserving separate 4B/9B result records.
- Debug the 4B proposal prompt for concrete-brief retention and JSON output; record actual NF4, right-context scoring, and production MCP validation.
- Record the later host-memory limitation affecting 9B reloads without overwriting earlier successful evidence.

## 0.1.0 — initial public version

- English phonetic comparison with positive soft rhyme weights and explanations.
- YAML/JSON aliases, direct vowel-pair rules, direction, syllable position, and pronunciation overrides.
- Dictionary search, contextual candidate ranking, and English lyric suggestions from English/Chinese briefs.
- Pinned Qwen3.5-9B/tokenizer, CUDA NF4 inference, and byte-preserving Windows resharding.
- Shared Python, JSON CLI, and five-tool stdio MCP interface.
- Measured local GPU and MCP smoke records; CPU core tests and packaging checks.

Current scope: candidate generation and reranking, English phonetics, symbolic rules. Audio input, Chinese phonetic scoring, and token-logit constraints are future work.
