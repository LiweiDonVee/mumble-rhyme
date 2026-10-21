"""Bounded token-tree search; language probabilities and rhyme preferences stay separate."""

import math
import re
from dataclasses import dataclass


def validate_search(beam_width=20, branch_top_k=20, max_new_tokens=12, min_new_tokens=2,
                    strength=2.0, length_normalization=0.0, limit=10, microbatch_size=1):
    for name, value, upper in (("beam_width", beam_width, 64), ("branch_top_k", branch_top_k, 100),
                               ("max_new_tokens", max_new_tokens, 48), ("min_new_tokens", min_new_tokens, 48),
                               ("limit", limit, 100), ("microbatch_size", microbatch_size, 8)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer between 1 and {upper}")
    if min_new_tokens > max_new_tokens:
        raise ValueError("min_new_tokens must not exceed max_new_tokens")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    if not math.isfinite(length_normalization) or not 0 <= length_normalization <= 1:
        raise ValueError("length_normalization must be between 0 and 1")


@dataclass(frozen=True)
class Path:
    ids: tuple = ()
    logprob: float = 0.0


def _boundary(text):
    """Only confirmed completed words guide an unfinished path.

    A decoded trailing word may still be a subword. A following space or
    punctuation confirms the earlier word; never pronounce partial BPE pieces.
    """
    match = re.match(r"^(.*?)(?:\s+[^\s]*|[.,!?;:\n]+)$", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def beam_search(next_tokens, decode, rhyme_weight, *, eos_ids=(), beam_width=20, branch_top_k=20,
                max_new_tokens=12, min_new_tokens=2, strength=2.0, length_normalization=0.0, limit=10):
    """Callbacks operate on token paths; next_tokens returns full-vocabulary log probabilities.

    Twenty active paths are independent from how the backend batches their
    forwards. One fifth of the frontier is reserved for raw language likelihood.
    """
    validate_search(beam_width, branch_top_k, max_new_tokens, min_new_tokens,
                    strength, length_normalization, limit)
    frontier, completed = [Path()], []
    expanded, pruned, forced, rounds = 0, 0, 0, 0
    rhyme_cache = {}

    def rhyme(text):
        if text not in rhyme_cache:
            weight = rhyme_weight(text)
            if not math.isfinite(weight) or not 0 < weight <= 1:
                raise ValueError("Rhyme callback must return a positive weight at most 1")
            rhyme_cache[text] = weight
        return rhyme_cache[text]

    def score(path, text):
        base = path.logprob / len(path.ids) ** length_normalization
        return base + strength * math.log(rhyme(text)) if text else base

    for step in range(max_new_tokens):
        if not frontier:
            break
        rounds += 1
        predictions = next_tokens([p.ids for p in frontier])
        if len(predictions) != len(frontier):
            raise ValueError("One next-token distribution is required per active path")
        children = []
        for parent, options in zip(frontier, predictions):
            for token, logprob in options[:branch_top_k]:
                if not math.isfinite(logprob) or logprob > 1e-6:
                    raise ValueError("Next-token log probabilities must be finite and nonpositive")
                expanded += 1
                path = Path(parent.ids + (token,), parent.logprob + logprob)
                raw = decode(path.ids)
                text = raw.strip()
                eos = token in eos_ids
                ended = eos or bool(re.search(r"[.!?\n]\s*$", raw))
                at_limit = step + 1 == max_new_tokens
                if ended or at_limit:
                    if len(path.ids) >= min_new_tokens and text and re.search(r"[A-Za-z]", text):
                        completed.append((path, raw, "eos" if eos else "punctuation" if ended else "max_tokens"))
                        forced += int(at_limit and not ended)
                    # Early EOS/punctuation cannot be extended into a later sentence.
                    continue
                children.append((path, _boundary(raw)))
        weighted = sorted(children, key=lambda item: (-score(*item), item[0].ids))
        language_slots = max(1, beam_width // 5) if strength else beam_width
        language = sorted(children, key=lambda item: (-item[0].logprob, item[0].ids))[:language_slots]
        selected = {p.ids: p for p, _ in language}
        for path, _ in weighted:
            if len(selected) >= beam_width:
                break
            selected.setdefault(path.ids, path)
        pruned += max(0, len(children) - len(selected))
        frontier = list(selected.values())

    rows = []
    for path, raw, finished in completed:
        text = raw.strip()
        rows.append({
            "text": text, "raw_continuation": raw, "token_ids": list(path.ids), "token_count": len(path.ids),
            "sequence_logprob": path.logprob, "sequence_probability": math.exp(path.logprob),
            "rhyme_weight": rhyme(text), "adjusted_log_weight": score(path, text),
            "finished_by": finished, "word_boundary_confirmed": finished != "max_tokens",
        })
    rows.sort(key=lambda r: (-r["adjusted_log_weight"], r["token_ids"]))
    unique = {}
    for row in rows:
        unique.setdefault(row["text"], row)
    result = list(unique.values())[:limit]
    if result:
        peak = result[0]["adjusted_log_weight"]
        denominator = sum(math.exp(r["adjusted_log_weight"] - peak) for r in result)
        for row in result:
            row["rank_weight"] = max(1e-300, math.exp(row["adjusted_log_weight"] - peak) / denominator)
    return {
        "candidates": result,
        "search": {
            "beam_width": beam_width, "branch_top_k": branch_top_k, "rounds": rounds,
            "expanded_edges": expanded, "pruned_paths": pruned, "completed_paths": len(completed),
            "duplicate_text_paths_collapsed": len(rows) - len(unique), "length_limit_reached": forced > 0,
            "forced_length_completions": forced, "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens, "exhaustive": False,
            "language_only_frontier_slots": max(1, beam_width // 5) if strength else beam_width,
        },
        "strength": strength, "length_normalization": length_normalization,
        "probability_definition": "Product of original full-vocabulary conditional token probabilities along "
        "the returned token path; includes EOS when present. Not normalized within top-k. Text alternatives "
        "are not marginalized over different token paths. rank_weight is relative to returned candidates only.",
    }
