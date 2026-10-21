# Changelog

## 0.1.0 — initial public version

- English phonetic comparison with positive soft rhyme weights and explanations.
- YAML/JSON aliases, direct vowel-pair rules, direction, syllable position, and pronunciation overrides.
- Dictionary search, contextual candidate ranking, and English lyric suggestions from English/Chinese briefs.
- Pinned Qwen3.5-9B/tokenizer, CUDA NF4 inference, and byte-preserving Windows resharding.
- Shared Python, JSON CLI, and five-tool stdio MCP interface.
- Measured local GPU and MCP smoke records; CPU core tests and packaging checks.

Current scope: candidate generation and reranking, English phonetics, symbolic rules. Audio input, Chinese phonetic scoring, and token-logit constraints are future work.
