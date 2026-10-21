# Design your own rhyme tolerances

[Home](../README.md) · [MCP & API](api.md)

A profile expresses how you intend to deliver the words. It assigns preferences, not permission to discard a phrase. All weights are positive: an unusual pair can stay in the running when its meaning works.

## Give a sound a memorable label

```yaml
name: my-delivery
aliases:
  i: {vowel_of: sit}
  e: {vowel_of: bed}
  ee: IY
vowel_rules:
  - sounds: [i, e]
    weight: 0.95
    symmetric: true
  - sounds: [i, e]
    weight: 0.98
    position: 1
syllables: 2
pronunciations:
  luv: [L AH1 V]
unknown_weight: 0.7
```

Save it as YAML and use `mumble-rhyme --profile your-profile.yaml compare sit set`. The included [mumble.yaml](../examples/mumble.yaml) is runnable.

`i` here means the vowel in *sit* (`IH`), not the letter name /aɪ/. The example alias uses the first dictionary reading and the final vowel; `syllable_from_end` can select an earlier vowel. You can use ARPAbet vowels directly: `AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW`.

## Rule precedence and direction

- `weight` is in `(0, 1]`. A value near 1 expresses strong acceptance. Zero is deliberately unavailable.
- `position: 1` means the final compared syllable; 2 means the preceding one. Position-specific rules override generic rules.
- At equal specificity, the last matching rule wins.
- `symmetric: true` is the default. With `false`, `[i, e]` means candidate `i` against reference `e` only.
- Pair rules are **not transitive**. Accepting `i/e` and `e/a` creates no automatic `i/a` rule.
- Unconfigured pairs use the default heuristic. They are not rejected.

Weights apply to vowels. Coda and stress still contribute to the final syllable weight. A configured vowel weight of 0.98 is therefore not a promise that the overall rhyme weight is exactly 0.98. The response exposes each component.

## Match the pronunciation you sing

`pronunciations` maps a word to one or more space-separated ARPAbet readings. Vowels can carry `0`, `1`, or `2` for stress. Use overrides for slang, names, a regional pronunciation, or a deliberate sung vowel. Overrides replace that word's dictionary alternatives; they do not infer pronunciation from audio.

Multiple readings are compared and the best phonetic pairing is returned. A phrase considers at most 32 pronunciation paths; the result reports when that search is truncated. Unknown words at the ending use `unknown_weight`. A known ending after an unknown interior word can still be compared, but the complete syllable count is unavailable.

## Tune the balance

| Field | Default | Meaning |
| --- | --- | --- |
| `syllables` | 1 | Number of final syllables to compare, 1–8 |
| `vowel_importance` | 0.75 | Vowel contribution; the remaining share is coda similarity |
| `stress_mismatch_weight` | 0.93 | Penalty for different stress |
| `unknown_weight` | 0.65 | Fallback when the ending pronunciation is unknown |
| `missing_syllable_weight` | 0.55 | Penalty for a missing compared syllable |
| `syllable_penalty` | 0.85 | Per-syllable distance penalty when a request sets `target_syllables` |

`strength` belongs to a ranking request, rather than a saved profile. It scales the log rhyme weight. Larger values prioritize rhyme more strongly; zero removes this term. `target_syllables` softly favors the length of the inserted text, not the entire surrounding line.

## Per-request profiles

MCP and service methods accept a `profile` object. Scalar settings replace the saved values, `aliases` and `pronunciations` merge by key, and `vowel_rules` replaces the complete rule list. The saved file is not edited.

```json
{
  "candidate": "sit",
  "target": "set",
  "profile": {
    "aliases": {"i": "IH", "e": "EH"},
    "vowel_rules": [{"sounds": ["i", "e"], "weight": 0.98}]
  }
}
```

A profile is applied during comparison and reranking. The current generator sees the reference and creative brief, but does not consume every custom pair rule. Bring your own candidates when the proposed set misses a sound you want.
