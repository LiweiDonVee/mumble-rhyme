import math

from .model import LocalModel
from .phonetics import RhymeEngine
from .profile import Profile


def finite_nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def validate_preferences(target, strength, target_syllables):
    if not target.strip():
        raise ValueError("Rhyme target must be nonempty")
    finite_nonnegative(strength, "strength")
    if target_syllables is not None and (not isinstance(target_syllables, int) or target_syllables < 1):
        raise ValueError("target_syllables must be a positive integer")


class LyricService:
    def __init__(self, profile: Profile | None = None, model: LocalModel | None = None):
        self.profile = profile or Profile()
        self.model = model or LocalModel()
        self.engine = RhymeEngine(self.profile)

    def _engine(self, patch=None):
        return RhymeEngine(self.profile.overlay(patch)) if patch else self.engine

    def compare(self, candidate, target, profile=None):
        return self._engine(profile).compare(candidate, target)

    def find(self, target, limit=20, profile=None):
        return self._engine(profile).find(target, limit)

    def rank(
        self,
        candidates,
        target,
        left="",
        right="",
        intent="",
        strength=2.0,
        target_syllables=None,
        use_model=True,
        profile=None,
        model_scores=None,
    ):
        if not 1 <= len(candidates) <= 32 or any(not isinstance(c, str) or not c.strip() for c in candidates):
            raise ValueError("Provide 1 to 32 nonempty candidate strings")
        validate_preferences(target, strength, target_syllables)
        engine = self._engine(profile)
        if model_scores is not None:
            if len(model_scores) != len(candidates) or any(not math.isfinite(s) for s in model_scores):
                raise ValueError("model_scores must contain one finite score for every candidate")
            lm = [{"mean_logprob": s, "sum_logprob": None, "scored_tokens": None} for s in model_scores]
            mode = "supplied_scores"
        elif use_model:
            lm = self.model.score(candidates, left, right, intent)
            mode = "local_model"
        else:
            lm = [{"mean_logprob": None, "sum_logprob": None, "scored_tokens": None} for _ in candidates]
            mode = "phonetic_only"
        rows = []
        for candidate, score in zip(candidates, lm):
            rhyme = engine.compare(candidate, target)
            syllables = rhyme["syllable_count"]
            penalty = (
                engine.profile.syllable_penalty ** abs(syllables - target_syllables)
                if target_syllables is not None and syllables is not None
                else 1.0
            )
            base = score["mean_logprob"] or 0.0
            adjusted = base + strength * math.log(rhyme["weight"]) + math.log(penalty)
            rows.append(
                {
                    "text": candidate,
                    "lm_mean_logprob": score["mean_logprob"],
                    "lm_sum_logprob": score["sum_logprob"],
                    "scored_tokens": score["scored_tokens"],
                    "rhyme": rhyme,
                    "syllable_weight": penalty,
                    "adjusted_score": adjusted,
                }
            )
        peak = max(r["adjusted_score"] for r in rows)
        weights = [max(1e-300, math.exp(r["adjusted_score"] - peak)) for r in rows]
        total = sum(weights)
        for row, weight in zip(rows, weights):
            row["rank_weight"] = weight / total
        rows.sort(key=lambda r: r["adjusted_score"], reverse=True)
        return {
            "candidates": rows,
            "scoring_mode": mode,
            "strength": strength,
            "profile": engine.profile.name,
            "model": self.model.status() if mode == "local_model" else None,
            "ranking_definition": "softmax(mean token log likelihood + strength * log(rhyme weight) "
            "+ log(syllable weight)); relative to this candidate set, not calibrated probabilities",
            "warnings": ["Pronunciation unknown: " + r["text"] for r in rows if r["rhyme"]["unknown_words"]],
        }

    def suggest(
        self, target, left="", right="", intent="", count=8, strength=2.0, target_syllables=None, profile=None
    ):
        validate_preferences(target, strength, target_syllables)
        # Validate profile before expensive GPU inference.
        self._engine(profile)
        candidates, raw = self.model.generate(left, right, intent, target, count)
        result = self.rank(candidates, target, left, right, intent, strength, target_syllables, True, profile)
        result["proposal_count_requested"] = count
        result["proposal_count_received"] = len(candidates)
        result["proposal_raw_output"] = raw
        result["search_scope"] = (
            "Bounded locally generated proposals; no rhyme filtering. "
            "Call rank_candidates with your own alternatives to widen the search."
        )
        return result
