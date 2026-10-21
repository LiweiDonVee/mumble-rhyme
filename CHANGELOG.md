# Changelog

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
