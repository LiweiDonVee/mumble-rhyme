from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

Weight = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VowelRule(StrictModel):
    sounds: tuple[str, str]
    weight: Weight
    symmetric: bool = True
    position: int | None = Field(default=None, ge=1, le=8)


class ExampleAlias(StrictModel):
    vowel_of: str
    syllable_from_end: int = Field(default=1, ge=1, le=8)


class Profile(StrictModel):
    name: str = "soft-default"
    aliases: dict[str, str | ExampleAlias] = Field(default_factory=dict)
    vowel_rules: list[VowelRule] = Field(default_factory=list)
    pronunciations: dict[str, list[str]] = Field(default_factory=dict)
    syllables: int = Field(default=1, ge=1, le=8)
    unknown_weight: Weight = 0.65
    missing_syllable_weight: Weight = 0.55
    vowel_importance: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    stress_mismatch_weight: Weight = 0.93
    syllable_penalty: Weight = 0.85

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def overlay(self, patch: dict | None) -> "Profile":
        # Scalars replace; mapping fields merge; rules replace to make precedence explicit.
        data = self.model_dump()
        for key, value in (patch or {}).items():
            if key in {"aliases", "pronunciations"}:
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return Profile.model_validate(data)
