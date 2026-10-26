"""Типизированный контракт: модель выбирает действия, но не координаты."""
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rect(StrictModel):
    x: int
    y: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


class WindowInfo(StrictModel):
    handle: int
    title: str
    process_name: str
    client: Rect


class Target(StrictModel):
    id: str
    rect: Rect
    confidence: float = Field(ge=0, le=1)
    frame_id: str

    @property
    def center(self) -> tuple[int, int]:
        return (self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2)


@dataclass(frozen=True)
class Observation:
    frame_id: str
    captured_at: float
    window: WindowInfo
    image: np.ndarray
    targets: dict[str, Target]
    ambiguous: frozenset[str] = frozenset()


class ExplainedAction(StrictModel):
    reason: str = Field(min_length=1, max_length=1500)

    @field_validator("reason")
    @classmethod
    def reject_control_characters(cls, value):
        if any(ord(char) < 32 or 127 <= ord(char) < 160 for char in value):
            raise ValueError("Объяснение не должно содержать управляющие символы")
        return value


class ClickAction(ExplainedAction):
    kind: Literal["click"] = "click"
    target_id: str = Field(min_length=1, max_length=128)


class KeyAction(ExplainedAction):
    kind: Literal["key"] = "key"
    key: Literal["esc"]


Action = Annotated[ClickAction | KeyAction, Field(discriminator="kind")]


class Plan(StrictModel):
    action: Action


class Step(StrictModel):
    id: str
    target_id: str
    expected_present: tuple[str, ...]
    expected_absent: tuple[str, ...] = ()
    explanation: str


class Verification(StrictModel):
    ok: bool
    detail: str
