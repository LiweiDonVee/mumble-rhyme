# CLI, MCP, and Python API

[Home](../README.md) · [Installation](installation.md) · [Rhyme rules](rhyme-rules.md)

## CLI

Global options come **before** the subcommand: `--profile path.yaml`, `--quantization nf4|none`.

| Command | Input | Output |
| --- | --- | --- |
| `compare TEXT TARGET` | A word or phrase and reference | Pronunciations, per-syllable explanation, soft weight |
| `find TARGET --limit 12` | Reference and 1–100 results | Softly ranked dictionary words |
| `rank --request request.json` | Supplied candidate list and context | All candidates with scores |
| `suggest --request request.json` | Creative brief and reference | Generated candidates with scores |
| `status` | None | Pinned model configuration; no weights loaded |
| `download` | None | Downloads the pinned snapshot |
| `serve` | stdio JSON-RPC | MCP server |

Normal CLI output is JSON on stdout. Errors are JSON on stderr with a nonzero exit code; model/dependency diagnostics also go to stderr. `--request -` reads a JSON object from stdin. Request files may have a UTF-8 BOM.

## Rank an existing phrase

```json
{
  "left": "I keep your voice inside my",
  "right": "",
  "target": "bed",
  "intent": "I keep hearing someone I miss. Conversational English.",
  "candidates": ["head", "chest", "dreams"],
  "strength": 2.0,
  "use_model": true
}
```

`left` and `right` surround the gap. Candidates contain only the inserted text. Leave both empty to compare complete lines. Phonetic scoring uses the **candidate's** ending, even when a `right` context follows it; language scoring includes the candidate and right context.

There must be 1–32 nonempty candidates. `use_model=false` explicitly chooses `phonetic_only` mode. Every supplied alternative is retained. Optional `target_syllables` is a positive integer length preference. Optional `profile` overrides the active profile for this request.

## Generate suggestions

```json
{
  "target": "night",
  "intent": "深夜失眠，想打电话却没勇气。口语化英文歌词。",
  "left": "",
  "right": "",
  "count": 6,
  "strength": 2.0,
  "target_syllables": 8
}
```

`count` is 1–12 and limits the proposal search. It does not enumerate possible lyrics. Proposals are sampled, so repeats of a request can differ. Empty context requests a complete line; context requests a gap fill. The response reports requested/received counts, raw model proposal text, and search scope.

## MCP setup

Use an absolute Python path so your client finds the correct environment. Replace **every example path** in [mcp.windows.json](../examples/mcp.windows.json) or [mcp.posix.json](../examples/mcp.posix.json). The JSON shows a common `mcpServers` layout; clients may use a different settings wrapper. No API key is required for local inference.

The server uses stdio, has no HTTP port, and starts through:

```console
python -m mumble_rhyme --profile examples/mumble.yaml serve
```

Registered tools:

| Tool | Required fields | Useful optional fields |
| --- | --- | --- |
| `compare_rhyme` | `candidate`, `target` | `profile` |
| `find_rhymes` | `target` | `limit` (default 20), `profile` |
| `rank_candidates` | `candidates`, `target` | `left`, `right`, `intent`, `strength`, `target_syllables`, `use_model`, `profile` |
| `suggest_lyrics` | `target` | `left`, `right`, `intent`, `count`, `strength`, `target_syllables`, `profile` |
| `model_status` | None | None |

Model loading is lazy. Keep the MCP process alive to reuse its model; allow sufficient time for the first call. Results are available as MCP structured content as well as text. Invalid requests return tool errors.

## Python

```python
from mumble_rhyme.profile import Profile
from mumble_rhyme.service import LyricService

service = LyricService(Profile.load("examples/mumble.yaml"))
comparison = service.compare("sit", "set")
ranked = service.rank(
    ["head", "chest", "dreams"], "bed",
    left="I keep your voice inside my", use_model=False,
)
# Loads Qwen only when a model operation is requested:
# suggestions = service.suggest("night", intent="Afraid to make the call", count=4)
```

The service shares the same logic as CLI and MCP. Advanced Python/CLI ranking can supply `model_scores`, one finite value per candidate, to combine externally obtained scores with the phonetic layer. This sets `scoring_mode="supplied_scores"`; it is not exposed by the MCP tool.

## Read a result

- `lm_mean_logprob`, `lm_sum_logprob`, `scored_tokens`: actual teacher-forced model scores; absent values in phonetic-only mode.
- `rhyme.weight`: a positive phonetic weight, plus selected phones, per-position alignment, rule source, unknown words and pronunciation-search coverage.
- `syllable_weight`: a separate soft meter preference.
- `adjusted_score`: model score plus log rhyme/meter adjustments.
- `rank_weight`: normalized over this candidate list only. Changing the list changes the weights.
- `model`: model ID, exact revision, quantization, and load state when model scoring is used.

These are selection aids, not calibrated probabilities of naturalness, musical fit, or listener preference. See [architecture](architecture.md) for the scoring boundaries.
