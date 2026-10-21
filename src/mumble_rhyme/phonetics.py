"""ARPAbet comparison. Numerical defaults are editable heuristics, not acoustic probabilities."""

import heapq
import math
import re
from functools import lru_cache
from itertools import islice, product

import cmudict

from .profile import ExampleAlias, Profile

# Approximate vowel height, backness and rounding; diphthongs carry their own positions.
VOWELS = {
    "IY": (0, 0, 0),
    "IH": (0.5, 0.3, 0),
    "EH": (1.5, 0, 0),
    "AE": (2.5, 0, 0),
    "AA": (3, 2, 0),
    "AH": (2, 1, 0),
    "AO": (2, 2, 1),
    "UH": (0.7, 1.7, 1),
    "UW": (0, 2, 1),
    "ER": (1.3, 1.2, 0),
    "EY": (1, 0, 0),
    "AY": (2.4, 0.6, 0),
    "OW": (1.2, 2, 1),
    "AW": (2.5, 1.4, 1),
    "OY": (1.8, 1.5, 1),
}
CONSONANTS = {
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
}
NEAR_CONSONANTS = [set(x.split()) for x in ["P B", "T D", "K G", "F V", "S Z", "SH ZH", "TH DH", "M N NG"]]
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


@lru_cache(maxsize=1)
def dictionary():
    return cmudict.dict()


def split_syllables(phones):
    result = []
    for phone in phones:
        base = phone.rstrip("012")
        if base in VOWELS:
            result.append({"vowel": base, "stress": int(phone[-1]) if phone[-1].isdigit() else 0, "coda": []})
        elif result:
            result[-1]["coda"].append(base)
    return result


def consonant_weight(left, right):
    # Weighted Levenshtein over codas; deletions and voicing changes remain possible.
    row = [float(i) for i in range(len(right) + 1)]
    for i, a in enumerate(left, 1):
        nxt = [float(i)]
        for j, b in enumerate(right, 1):
            cost = 0 if a == b else (0.3 if any(a in s and b in s for s in NEAR_CONSONANTS) else 0.8)
            nxt.append(min(row[j] + 1, nxt[j - 1] + 1, row[j - 1] + cost))
        row = nxt
    return math.exp(-0.7 * row[-1] / max(1, len(left), len(right)))


class RhymeEngine:
    def __init__(self, profile: Profile | None = None):
        self.profile = profile or Profile()
        self.variants = lru_cache(maxsize=8192)(self._variants)
        self.overrides = {}
        for word, readings in self.profile.pronunciations.items():
            if not readings:
                raise ValueError(f"No pronunciations provided for {word!r}")
            for reading in readings:
                phones = reading.split()
                if not phones or not any(p.rstrip("012") in VOWELS for p in phones):
                    raise ValueError(f"Pronunciation needs a vowel: {word!r}")
                if any(
                    not re.fullmatch(r"(?:" + "|".join(VOWELS) + r")[012]?", p) and p not in CONSONANTS
                    for p in phones
                ):
                    raise ValueError(f"Invalid ARPAbet pronunciation: {reading!r}")
            self.overrides[word.lower()] = [x.split() for x in readings]
        self.aliases = {}
        for name, alias in self.profile.aliases.items():
            if isinstance(alias, ExampleAlias):
                readings = self.readings(alias.vowel_of)
                if not readings:
                    raise ValueError(f"Alias example has no pronunciation: {alias.vowel_of!r}")
                syllables = split_syllables(readings[0])
                if len(syllables) < alias.syllable_from_end:
                    raise ValueError(f"Alias position exceeds syllables: {name!r}")
                value = syllables[-alias.syllable_from_end]["vowel"]
            else:
                value = alias.upper()
            if value not in VOWELS:
                raise ValueError(f"Alias {name!r} must resolve to an ARPAbet vowel")
            self.aliases[name] = value
        self.rules = []
        for rule in self.profile.vowel_rules:
            sounds = tuple(self.aliases.get(s, s.upper()) for s in rule.sounds)
            if any(s not in VOWELS for s in sounds):
                raise ValueError(f"Unknown vowel or alias in rule: {rule.sounds}")
            self.rules.append((sounds, rule))

    def readings(self, word):
        return self.overrides.get(word.lower(), dictionary().get(word.lower(), []))

    def vowel_weight(self, candidate, target, position):
        matches = []
        for order, (sounds, rule) in enumerate(self.rules):
            if rule.position is not None and rule.position != position:
                continue
            if sounds == (candidate, target) or (rule.symmetric and sounds == (target, candidate)):
                matches.append((rule.position is not None, order, rule))
        if matches:
            rule = max(matches, key=lambda x: x[:2])[2]
            return rule.weight, "custom_rule"
        if candidate == target:
            return 1.0, "identical_vowel"
        a, b = VOWELS[candidate], VOWELS[target]
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        return max(0.3, math.exp(-0.23 * distance)), "default_similarity"

    def _variants(self, text):
        words = WORD.findall(text.lower().replace("’", "'"))
        unknown = [w for w in words if not self.readings(w)]
        # An unknown at the end prevents inventing the sound of that ending.
        # An interior unknown marks a boundary; known trailing words can still rhyme.
        tail = []
        for word in reversed(words):
            readings = self.readings(word)
            if not readings:
                break
            tail.insert(0, readings)
        count = math.prod(len(x) for x in tail) if tail else 0
        variants = []
        for combination in islice(product(*tail), 32) if tail else []:
            phones = [p for reading in combination for p in reading]
            variants.append((phones, split_syllables(phones)))
        return variants, list(dict.fromkeys(unknown)), count > 32

    def compare(self, candidate: str, target: str) -> dict:
        if not candidate.strip() or not target.strip():
            raise ValueError("Candidate and rhyme target must be nonempty")
        left, unknown_l, truncated_l = self.variants(candidate)
        right, unknown_r, truncated_r = self.variants(target)
        common = {
            "candidate": candidate,
            "target": target,
            "unknown_words": list(dict.fromkeys(unknown_l + unknown_r)),
            "syllable_count": None,
            "pronunciation_search_truncated": truncated_l or truncated_r,
            "pronunciation_pairs_considered": len(left) * len(right),
        }
        if not left or not right:
            return {
                **common,
                "weight": self.profile.unknown_weight,
                "alignment": [],
                "candidate_phones": [],
                "target_phones": [],
                "status": "unknown_pronunciation",
            }
        best = None
        for (lp, ls), (rp, rs) in product(left, right):
            alignment, weights = [], []
            for pos in range(1, self.profile.syllables + 1):
                if len(ls) < pos or len(rs) < pos:
                    weight = self.profile.missing_syllable_weight
                    alignment.append(
                        {"position_from_end": pos, "weight": weight, "source": "missing_syllable"}
                    )
                else:
                    a, b = ls[-pos], rs[-pos]
                    vowel, source = self.vowel_weight(a["vowel"], b["vowel"], pos)
                    coda = consonant_weight(a["coda"], b["coda"])
                    stress = 1 if a["stress"] == b["stress"] else self.profile.stress_mismatch_weight
                    importance = self.profile.vowel_importance
                    weight = vowel**importance * coda ** (1 - importance) * stress
                    alignment.append(
                        {
                            "position_from_end": pos,
                            "candidate": a,
                            "target": b,
                            "vowel_weight": vowel,
                            "coda_weight": coda,
                            "stress_weight": stress,
                            "weight": weight,
                            "source": source,
                        }
                    )
                weights.append(weight)
            score = math.exp(sum(math.log(w) for w in weights) / len(weights))
            if best is None or score > best["weight"]:
                best = {
                    "weight": score,
                    "alignment": alignment,
                    "candidate_phones": lp,
                    "target_phones": rp,
                    "syllable_count": len(ls) if not unknown_l else None,
                    "status": "scored",
                }
        return {**common, **best}

    def find(self, target: str, limit: int = 20, mode="balanced", commonness_strength=0.65,
             collapse_homophones=None):
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not target.strip():
            raise ValueError("Rhyme target must be nonempty")
        if mode not in {"balanced", "phonetic"}:
            raise ValueError("mode must be balanced or phonetic")
        if not math.isfinite(commonness_strength) or commonness_strength < 0:
            raise ValueError("commonness_strength must be finite and nonnegative")
        from wordfreq import zipf_frequency

        collapse = mode == "balanced" if collapse_homophones is None else collapse_homophones
        vocabulary = (w for w in dictionary() if WORD.fullmatch(w) and w.lower() != target.lower())
        count = 0

        def scored():
            nonlocal count
            for word in vocabulary:
                count += 1
                rhyme = self.compare(word, target)
                frequency = zipf_frequency(word, "en")
                commonness = max(0.02, min(1.0, frequency / 7))
                score = rhyme["weight"] if mode == "phonetic" else (
                    rhyme["weight"] ** 2 * commonness ** commonness_strength
                )
                yield {"text": word, "rhyme": rhyme, "frequency_zipf": frequency,
                       "commonness_weight": commonness, "lookup_score": score, "alternatives": []}

        rows = scored()
        if collapse:
            groups = {}
            for row in rows:
                key = tuple(row["rhyme"].get("candidate_phones", [])) or (row["text"],)
                groups.setdefault(key, []).append(row)
            representatives = []
            for group in groups.values():
                group.sort(key=lambda r: (-r["lookup_score"], r["text"]))
                best = group[0]
                best["alternatives"] = [
                    {"text": r["text"], "frequency_zipf": r["frequency_zipf"],
                     "lookup_score": r["lookup_score"]} for r in group[1:]
                ]
                representatives.append(best)
            rows = representatives
        items = heapq.nsmallest(limit, rows, key=lambda r: (-r["lookup_score"], r["text"]))
        return {
            "candidates": items,
            "vocabulary_evaluated": count,
            "search_scope": "CMUdict single words; all evaluated with soft weights; top results returned",
            "profile": self.profile.name,
            "mode": mode,
            "commonness_strength": commonness_strength if mode == "balanced" else 0,
            "homophones_collapsed": collapse,
            "ranking_definition": "rhyme^2 * max(0.02, min(1, English Zipf frequency / 7))^commonness_strength; "
            "phonetic mode uses rhyme weight only; frequency never changes the rhyme component",
        }
