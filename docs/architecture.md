# Architecture and scoring boundaries

[Home](../README.md) · [Rhyme rules](rhyme-rules.md)

Version 0.3 exposes pronunciation, candidate ranking, and bounded token-tree search. It does not modify the model's token distribution at each decoding step.

```mermaid
flowchart TD
    A[Meaning and left/right context] --> B[Pinned local Qwen proposals]
    C[Your own candidate phrases] --> D[Candidate pool]
    B --> D
    D --> E[Full-text tokenizer and sequence likelihood]
    D --> F[Dictionary and pronunciation overrides]
    G[User rhyme profile] --> F
    E --> H[Soft rhyme and syllable adjustment]
    F --> H
    H --> I[All supplied candidates with explanations]
```

## Modules

| Module | Responsibility |
| --- | --- |
| `profile.py` | Validate YAML/JSON rules and request overlays |
| `phonetics.py` | Pronunciation alternatives, vowel/coda/stress comparison, dictionary search |
| `model.py` | Pinned tokenizer/model, proposals, sequence likelihood, next-token distributions |
| `search.py` | Backend-independent beam frontier, completed-word rhyme guidance, raw path probability and cutoff reporting |
| `checkpoint.py` | Byte-preserving Windows resharding to lower file-mapping peaks |
| `service.py` | Input validation and combined candidate ranking |
| `cli.py`, `server.py` | JSON CLI and stdio MCP over the same service |

## Phonetic layer

English readings come from CMUdict plus user overrides. A heuristic separates vowel nuclei and following consonants, compares the final N syllables, and combines vowel, coda, and stress weights. The default similarity is hand-designed; it is not a learned singing perception score. Explicit pair rules override the relevant vowel comparisons without building equivalence classes.

The best pronunciation pairing is returned, bounded to 32 paths per phrase. Unknown endings have an explicit positive fallback. Unknown interior words can leave a comparable known ending while making the complete syllable count unavailable. The response discloses these cases. All supplied candidates survive rhyme evaluation.

## Language layer and the fixed vocabulary

The selected Qwen weights and tokenizer use the same immutable revision. The model registry pins 9B and 4B separately; default 9B is preserved and model selection never falls back to a different size. Candidate phrases are not assumed to be single tokens. The adapter assembles prompt + left context + candidate + right context, tokenizes the complete string once, and selects token positions using character offsets. A token spanning the left/gap boundary is scored. Right context is also scored to account for the connection after the gap.

Scores are teacher-forced token log likelihoods. Ranking uses their mean to reduce raw sequence-length bias; this does not remove all length effects. The raw sum and number of scored tokens remain available. Model input is capped at 2,048 tokens with an explicit error on overflow. Selected output positions avoid materializing vocabulary logits for the entire prompt.

The Qwen proposal prompt asks for English JSON strings and a rhyme reference. Custom phonetic profiles currently affect the reranker, not the generation distribution. This finite proposal stage can miss an unconventional rhyme; supplying candidates is an important supported path.

## Combined ranking

```text
s(c) = mean_log_likelihood(c and right context | prompt and left context)
     + strength × log(rhyme_weight(c))
     + log(syllable_weight(c))
rank_weight(c) = softmax(s(c), over this candidate set)
```

When `use_model=false`, the language term is absent and mode is explicitly `phonetic_only`. A zero rhyme strength leaves language and optional syllable preferences. The phonetic score does not act as a hard gate. Stable normalization keeps ranking weights positive.

## Local execution

Model imports and weights are lazy; a shared process-wide project lock serializes all LocalModel loading, inference, and explicit `LocalModel.unload()`. Failed loads clear allocation-owning exception frames, including chained causes/contexts, and close readers. The server retains the loaded model for a writing session. Downloads use Hugging Face; after complete caching, `HF_HUB_OFFLINE=1` permits local operation.

Windows uses standard smaller safetensors shards, copied by streaming raw tensor bytes. Validated large NF4 tensors are transferred directly to CUDA using a 16 MiB CPU buffer; smaller tensors use the single-buffer full-tensor reader. Default unload drops this adapter's owned references and unused CUDA cache. Explicit `unload(release_runtime_cache=True)` also synchronizes and clears the pinned PyTorch version's process-wide cached cuBLAS workspaces; it requires external CUDA work in the same process to be stopped. Cache preparation is locked and publishes its completed index last. The source snapshot remains untouched. The local integration record includes a comparison of all 775 tensor hashes. Windows also initializes optional NumPy before starting stdio reader threads to avoid the observed extension-import stall.

## Deliberate limits

English pronunciations only; Chinese text is creative intent. No audio transcription, automatic sung-vowel inference, accent model, artist corpus, or constrained-logit backend. This release makes user preferences explicit and inspectable while leaving final writing decisions with the creator.

Before NF4 weights are loaded, the current execution thread initializes cuBLAS and performs a small BF16 linear operation. The MCP SDK's synchronous tools execute on that thread. Allocation failures in scoring/generation are handled after their frames unwind, then weights and temporary references are released. Runtime/validation errors unrelated to allocation retain their normal behavior; there is no hidden retry or model substitution. See [cold suggestion and recovery checks](suggest-reliability.md).

## Direct probability search

`search_lyrics` branches actual token IDs with original full-vocabulary log probabilities, removes nontext control tokens after normalization, and preserves EOS probability. Active paths are pruned using completed-prefix rhyme guidance with a fraction reserved for raw likelihood. Final candidates retain raw path probability plus separate soft weighting; no Qwen self-rating is used. Microbatching is independent from beam width, and full-prefix forwards avoid persistent per-beam KV copies. See [probability-search.md](probability-search.md).

Windows loading additionally scopes a thread-local TorchFunctionMode to the pinned Transformers `caching_allocator_warmup` frame, limiting its disposable torch.empty to 256 MiB. Ordinary parameter/activation allocations are forwarded unchanged, and the mode exits on success/error without mutating module globals. See [allocation audit](windows-allocation.md).
