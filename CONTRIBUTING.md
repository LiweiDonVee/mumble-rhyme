# Contributing

Contributions are welcome in English or Chinese. Useful contributions include pronunciation examples, clear bug reports, documentation fixes, and bounded improvements to candidate coverage.

## Development

From a clone, create and activate a Python 3.10+ environment, then:

```console
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests scripts
python -m build
```

Core tests do not download Qwen and do not require torch or a GPU. They include an actual MCP stdio subprocess. CI runs the core on Windows and Linux; it does not certify GPU compatibility.

To reproduce the separately measured model path, install the model extra using [the tested CUDA instructions](docs/installation.md), download the pinned model, and opt in:

```console
python scripts/check_model_adapter.py
python scripts/smoke_model.py
python scripts/smoke_mcp_model.py
```

The adapter check uses the real tokenizer and a random tiny model; it is not a pretrained-model quality test. All three scripts accept `--model 4b` or `--model 9b` (the default). The latter two scripts use the selected actual model and write JSON under `docs/`, with a `-4b` suffix for 4B so recorded 9B results are preserved. Generation is sampled, so text and runtime can differ. Do not replace recorded output with invented examples.

## Changes and tests

Keep CLI, Python, and MCP behavior aligned through the shared service. Add meaningful regression tests for changed behavior. Preserve positive phonetic weights, retention of every supplied candidate, rule direction and precedence, and the fixed tokenizer/model revision unless the change explicitly revises that contract. A future model upgrade needs evidence for both tokenization and actual inference.

For pronunciation reports, include:

1. A short line or gap context you may share publicly.
2. Candidate/reference words and the reading you intend, using example words or ARPAbet.
3. Your profile and the expected difference in ranking or explanation.
4. The actual result and whether it used local-model or phonetic-only scoring.

For inference reports, include OS, Python, GPU, torch/CUDA, model revision, and an error traceback with private paths/text removed. Open an issue before a large backend, audio, or language-support change so the scope is clear.

## Public repository boundaries

Do not commit model weights, caches, virtual environments, credentials, private songs, or logs containing unpublished writing. MCP examples use intentionally replaceable absolute paths. Internal planning notes and local research are excluded from distribution.

Contributions are provided under GPL-3.0-or-later, the same license as this project. Third-party material must retain its applicable license and attribution. No contributor agreement is required.

Windows/CUDA memory regression audit (real weights, optional): `python scripts/check_model_memory.py --model 9b` and `--model 4b`. It injects a partial-load failure and checks recovery plus two load/score/unload cycles. Run heavyweight GPU checks serially on constrained hosts. Functional smoke scripts accept `--output path.json` to preserve older records. `tests/test_tensor_stream.py` uses real torch tensors when the model runtime is installed and is skipped in core-only CI.
