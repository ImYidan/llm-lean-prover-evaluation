"""Generation candidate boundary shared by model backends."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    text: str
    finish_reason: str | None = None
    stop_reason: str | int | None = None
