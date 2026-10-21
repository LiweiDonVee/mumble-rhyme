# Mumble Rhyme

**Your rhyme rules. Your words. More ways to land the line.**

[中文说明](README.zh-CN.md) · [Installation](docs/installation.md) · [Rhyme rules](docs/rhyme-rules.md) · [MCP & API](docs/api.md)

A local English songwriting toolkit for writers who hear a line before they can phrase it. Describe the meaning in English or Chinese, fill a gap, or compare complete lines. Decide which sounds can rhyme in your delivery—and how much each compromise should cost.

Use it from a terminal, Python, or any client that supports **stdio MCP**. Phonetic tools run without a language model. Generation and contextual scoring use pinned **Qwen3.5-9B or Qwen3.5-4B** weights with the selected model's matching tokenizer. The default remains 9B; choose 4B explicitly for a smaller model.

## Start with the sound you mean

Maybe `sit` and `set` work together in the way you sing. Make that a reusable preference:

```yaml
aliases:
  i: {vowel_of: sit}
  e: {vowel_of: bed}
vowel_rules:
  - sounds: [i, e]
    weight: 0.98
    symmetric: true
```

```console
mumble-rhyme --profile examples/mumble.yaml compare sit set
```

The response shows the selected pronunciations, the `custom_rule` that matched, and a vowel weight of `0.98`. An unconfigured pair still gets a positive default similarity. An uncertain pronunciation stays visible.

The labels `i` and `e` are yours: example words define their sounds. Profiles support directional pair rules, syllable-specific preferences, and sung pronunciation overrides. Accepting two vowel pairs does not silently create a third.

## A writing session, three useful moves

| When you get stuck | Give the tool | Get back |
| --- | --- | --- |
| “I know what I mean, but not how to say it.” | A meaning, a rhyme reference, optional surrounding words | English gap fills or whole lines, then soft ranking |
| “These lines mean roughly the same thing.” | Your alternatives and both sides of the gap | Contextual model scores, rhyme explanations, a ranked list |
| “This sound works in my delivery.” | Example words, a weight, optional pronunciation overrides | A saved profile you can reuse or pass to another model |

A rhyme lookup is useful for exploring vocabulary. A writing assistant is useful for developing an idea. Mumble Rhyme adds a small, inspectable layer between them: **bring your own phrases and tell the ranking what you can sing.** See [how it fits different workflows](docs/workflows.md).

## Try it without a GPU

From a clone of this repository, using Python 3.10+:

```console
git clone https://github.com/LiweiDonVee/mumble-rhyme.git
cd mumble-rhyme
python -m venv .venv
```

Activate it: `.venv\Scripts\Activate.ps1` on Windows, or `source .venv/bin/activate` on Linux/macOS. Then:

```console
python -m pip install -e .
mumble-rhyme --profile examples/mumble.yaml compare sit set
mumble-rhyme find night --limit 12
mumble-rhyme status
```

No model download is needed for these commands. The package is installed from this repository; a PyPI release is not required.

## Add local generation and context scoring

The tested GPU setup is **Windows, Python 3.10, RTX 5080, PyTorch 2.8.0+cu128**:

```console
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[model]"
mumble-rhyme download
mumble-rhyme --profile examples/mumble.yaml rank --request examples/rank.json
mumble-rhyme --profile examples/mumble.yaml suggest --request examples/suggest.json
```

Original weights need about **19.3 GB** of disk. On Windows, the first load creates an additional **19.3 GB** of smaller shards to reduce temporary mapping pressure; tensor values are unchanged. The tested NF4 run peaked at **7.525 GiB of PyTorch allocated GPU memory**. This is a measurement, not a minimum VRAM guarantee. See [hardware, cache, and troubleshooting](docs/installation.md).

Choose 4B with a global option before the subcommand:

```console
mumble-rhyme --model 4b download
mumble-rhyme --model 4b status
mumble-rhyme --model 4b --profile examples/mumble.yaml suggest --request examples/suggest.json
```

| Selection | Official weights and tokenizer | Fixed revision | Weight download |
| --- | --- | --- | --- |
| `--model 9b` (default) | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 19.31 GB |
| `--model 4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9.32 GB |

Each selection locks weights and tokenizer together. Models are never silently substituted. The default NF4 path requires CUDA; core phonetic tools do not. Weights are downloaded separately under their upstream license.

## Give your assistant a rhyme tool

Start the stdio MCP server:

```console
mumble-rhyme --profile examples/mumble.yaml serve
```

Copy [the Windows configuration](examples/mcp.windows.json) or [the POSIX configuration](examples/mcp.posix.json) into your MCP client's server settings, replacing the example absolute paths.

For 4B use [the Windows 4B configuration](examples/mcp.windows.4b.json) or [the POSIX 4B configuration](examples/mcp.posix.4b.json), or start `mumble-rhyme --model 4b --profile examples/mumble.yaml serve`. A server keeps one model selection for its lifetime.

> Help me express “I'm awake, staring at their number, but I can't make the call.” Use `night` as a rhyme reference. Suggest several lines, explain the English, then compare the ones I like using my rhyme profile. Keep unusual near rhymes available.

| Tool | Purpose | Needs Qwen |
| --- | --- | --- |
| `compare_rhyme` | Explain a word/phrase rhyme | No |
| `find_rhymes` | Search dictionary words by soft phonetics | No |
| `rank_candidates` | Compare your phrases in context | Default yes; `use_model=false` available |
| `suggest_lyrics` | Generate English options and rerank them | Yes |
| `model_status` | Inspect pinned configuration and load state | No |

The server reuses the loaded model. Separate CLI invocations load it again. Python callers can explicitly release weights with `service.model.unload()`; see the [memory lifetime checks](docs/model-memory.md). Full request examples and Python usage are in [MCP & API](docs/api.md).

## How the ranking works

```text
meaning + context → Qwen proposals (or your own alternatives)
                           ↓
contextual sequence likelihood + pronunciation alternatives + your rhyme profile
                           ↓
                 explained, softly ranked candidates
```

```text
score = mean_token_log_likelihood
      + strength * log(rhyme_weight)
      + log(syllable_weight)
```

`strength=0` removes the rhyme adjustment; any separate syllable preference still applies. Every supplied candidate remains. `rank_weight` is a softmax within that candidate set, not a probability that a line will sound good.

Version 0.2 implements **candidate generation and reranking**, rather than modifying token logits during generation. The proposal prompt uses the rhyme reference; custom pair rules take effect during reranking. You can supply your own alternatives to explore possibilities the generator missed. Complete text is tokenized before scoring, including right context and tokens that cross the gap boundary. See [the implementation](docs/architecture.md).

## Reliability and verification

Version 0.2.2 covers the reported cold `suggest_lyrics` cuBLAS allocation case: CUDA runtime initialization happens before NF4 loading, inference allocation failures release weights, and status reports the running package version. Both models have real suggest-first and same-process failure-recovery records. See [the investigation, limits, and client restart instructions](docs/suggest-reliability.md).

## What is verified, and what is next

- **Verified locally:** 50 core tests plus 10 optional tensor tests, a real MCP stdio exchange, pinned Qwen3.5-9B and 4B NF4 generation/scoring, and unchanged tensors after Windows resharding (775 for 9B, 738 for 4B).
- **Measured 4B:** 12.56 seconds for a cached direct-model smoke, 14.68 seconds for the production MCP exchange, peak PyTorch allocation 3.217 GiB.
- **Recorded 9B:** one cached direct-model smoke took 27.71 seconds; an MCP smoke including model load, ranking, and generation took 26.35 seconds. These are functional checks with different requests, not comparative benchmarks or generation-only latency.
- **Current scope:** English pronunciations; English or Chinese creative instructions; JSON CLI, Python, and MCP.
- **Future work:** better proposal coverage for custom profiles, broader pronunciation handling, Chinese phonetic scoring, and evaluation with real writing sessions.

No recording analysis, melody tracking, browser editor, or trained singing-quality score is included. Default phonetic similarity is a heuristic over dictionary pronunciations. Read the [verification record](docs/verification.md) for raw outputs and limits, including the [0.2.1 memory audit and resolved reload failure](docs/model-memory.md).

## Contribute and license

Share a short context, the pronunciation you intend to sing, and the alternatives you expect to rank differently. Those examples are more useful than a bare “this doesn't rhyme.” Development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

Code: **[GPL-3.0-or-later](LICENSE)**. The current Python pronunciation wrapper uses the same license. Dictionary data and Qwen weights retain their own upstream terms; see [third-party notices](THIRD_PARTY_NOTICES.md). No model weights or artist lyric corpus are bundled.
