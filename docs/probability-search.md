# Token-path search and practical rhyme lookup

Version 0.3.0 adds `search_lyrics`, a probability-based tree search. It does not ask Qwen to rate semantic fit, naturalness, or lyric quality. Qwen supplies actual next-token probabilities conditioned on the meaning and lyric prefix; the pronunciation engine supplies your soft rhyme preferences.

## Try it

CLI request: [examples/search.json](../examples/search.json).

```console
mumble-rhyme --model 4b --profile examples/mumble.yaml search --request examples/search.json
```

MCP tool arguments use the same fields:

```json
{
  "target": "light",
  "left": "the city lights are",
  "intent": "深夜城市的孤独感",
  "beam_width": 20,
  "branch_top_k": 20,
  "min_new_tokens": 2,
  "max_new_tokens": 12,
  "strength": 2.0,
  "length_normalization": 0.0,
  "limit": 10,
  "microbatch_size": 1
}
```

Python: `service.search("light", left="the city lights are", intent="深夜城市的孤独感")`.

## What the tree means

Each edge adds a token with the original full-vocabulary conditional probability. At each depth, each active path expands its top 20 eligible text/EOS tokens; at most 20 paths continue. Four default frontier slots preserve pure language likelihood, with the remainder selected using rhyme guidance. This is a finite beam search, not exhaustive enumeration or guaranteed global optimization.

Nontext control tokens are excluded **after** log-softmax. The remaining top-k probabilities are never renormalized. EOS contributes its probability when a path ends with EOS. Punctuation/newline can also terminate a path. A path that stops too early is not continued into a second sentence. Special-token exclusion and pruning mean some possible continuations are not explored.

Subword pieces are not pronounced as words. For unfinished paths, only the prefix confirmed by a following space or punctuation receives rhyme guidance. The next word may still be partial. Final punctuation or EOS confirms an ending; reaching the token limit does not, and that result is marked `finished_by="max_tokens"`, `word_boundary_confirmed=false`. Unknown or uncertain endings keep positive weights.

## Read the probabilities

```text
sequence_logprob = sum(log P_Qwen(next token | prompt, left, earlier path tokens))
sequence_probability = exp(sequence_logprob)
adjusted_log_weight = sequence_logprob / token_count^length_normalization
                    + strength * log(rhyme_weight)
```

- `sequence_logprob` is authoritative when the probability product is too small for a floating-point number. It includes every returned token ID, including EOS if emitted.
- `raw_continuation` and `token_ids` identify the actual path; `text` is its stripped display form. Whitespace and a stop token still contribute to probability.
- `conditioning_token_ids` records the exact chat/intent/prefix tokens for independent verification with the pinned model.
- `rhyme_weight` and the detailed `rhyme` explanation are separate from the model probability. No frequency prior or subjective self-rating is added to tree search.
- `rank_weight` is normalized only over the returned candidates. It is not the original probability mass of the language model or a probability of musical quality.
- When different token paths display the same text, the best-ranked path is retained; their probabilities are not summed into a marginal probability for the string.

The default `length_normalization=0` preserves the raw product before rhyme weighting. This naturally favors short expressions: one recorded 4B result is `just`. Increase `min_new_tokens` to explore a longer range, or explicitly set normalization between 0 and 1. That changes the ranking heuristic, not the returned raw probability. Token counts include punctuation and EOS, not just English words.

There is no right-context input in this release's forward search. Use `rank_candidates` to compare supplied infills against both sides of a gap. Chinese intent is accepted, while pronunciation/rhyme scoring remains English.

## Width, batching, and limits

| Setting | Default | Allowed |
| --- | --- | --- |
| `beam_width` | 20 active paths | 1–64 |
| `branch_top_k` | 20 extensions per path | 1–100 |
| `min_new_tokens` / `max_new_tokens` | 2 / 12 | 1–48, min no larger than max |
| `limit` | 10 returned candidates | 1–100 |
| `microbatch_size` | 1 model forward at a time | 1–8 |
| `length_normalization` | 0 | 0–1 |
| `strength` | 2 | finite and nonnegative |

Search width and batch size are independent: microbatch 1 still evaluates 20 active paths. The backend recomputes complete prefixes, without retaining 20 KV-cache copies. This trades speed for bounded memory and avoids unsupported cache branching in Qwen's hybrid attention. Input plus token budget must fit the existing 2,048-token cap.

Quantized BF16 computation can change slightly with batch shape; increasing microbatch size can alter low-ranked token probabilities and eventually the beam. A measured 9B top-token logprob changed by about 0.123 between batch sizes 1 and 2. The raw records disclose that numeric drift; bitwise equality across batches is not claimed.

Results disclose rounds, expanded edges, pruned paths, completed paths, collapsed duplicates, and whether the length budget was reached. No rhyme is banned, but top-k selection, beam pruning, termination rules, and output limits restrict the search. Low-probability rhyme families can still be missed. Increase budgets or submit your own phrases to `rank_candidates` when needed.

## More useful dictionary exploration

`find_rhymes` now defaults to `mode="balanced"`:

```text
commonness = max(0.02, min(1, English Zipf frequency / 7))
lookup_score = rhyme_weight^2 * commonness^commonness_strength
```

The default commonness strength is 0.65. The score is a lookup preference, not a probability. Low frequency lowers a word's position without changing its pronunciation score or banning it. `commonness_strength=0` disables its influence. Words sharing the selected phoneme sequence are grouped; the representative includes expandable `alternatives` with their spellings, frequency, and lookup score.

`mode="phonetic"` uses only rhyme weight and preserves separate spellings by default. `collapse_homophones` can explicitly enable/disable grouping in either mode. CLI: `mumble-rhyme find mine --mode phonetic`.

In the local balanced run, `mine` includes `line`, `time`, `fine`, `design`, `sign`, `wine`; `fall` includes `all`, `call`, `small`, `ball`, `wall`, `hall`. Names and Internet vocabulary can remain: general word frequency is not a songwriting-specific word classifier. Same-pronunciation words can have different meanings, so variants remain visible.

The full dictionary is still evaluated, with its budget reported. Lookup uses installed `wordfreq` English data locally; it does not start Qwen, browse, or send a query to a frequency service. See [third-party notices](../THIRD_PARTY_NOTICES.md) for the distinct code/data licenses.

## Verification

`scripts/check_search_model.py --model 4b` (or `9b`) runs the default full beam, then independently recomputes token log likelihoods for returned paths. It also records single/batched numeric differences. `scripts/check_search_mcp.py` verifies the cold production tool with the same 20×20, 12-token budget.

Records: [4B model](search-model-4b.json), [9B model](search-model-9b.json), [4B MCP](search-mcp-4b.json), [9B MCP](search-mcp-9b.json). These are functional/probability checks, not a claim that beam search always writes better lines or finds the best rhyme.
