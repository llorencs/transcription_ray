# Copy the original segmenter models to maintain compatibility
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field


def initialize_punctuation() -> set[str]:
    """Initialize a set of common punctuation marks used in various languages."""
    import string

    western_punctuation = set(string.punctuation)
    japanese_punctuation = {"。", "、", "！", "？", "；", "：", "（", "）", "〜", "ー"}
    chinese_punctuation = {
        "。",
        "，",
        "！",
        "？",
        "；",
        "：",
        "（",
        "）",
        """, """,
        "'",
        "'",
        "——",
        "……",
    }
    arabic_punctuation = {"،", "؛", "؟", "۔"}

    return (
        western_punctuation
        | japanese_punctuation
        | chinese_punctuation
        | arabic_punctuation
    )


class SegmenterConfig(BaseModel):
    max_chars_per_line: Annotated[int, Field(default=42)]
    max_chars_per_subtitle: Annotated[int, Field(default=84)]
    max_duration: Annotated[float, Field(default=5.0)]
    pause_threshold: Annotated[float, Field(default=0.8)]
    punctuation_marks: Annotated[
        List[str], Field(default_factory=lambda: [".", "!", "?", "。", "！", "？", "؟"])
    ]
    end_on_punctuation: Annotated[bool, Field(default=True)]


class SubtitleSegmentModel(BaseModel):
    start: float
    end: float
    lines: List[str]


# Re-export EventModel from pydantic_models to avoid duplication
from .pydantic_models import EventModel, WordModel
