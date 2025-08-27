# Enhanced version of the original segmenter with additional features
from typing import List, Optional, Annotated, Tuple
from pydantic import BaseModel, Field
import string

from src.models.segmenter_models import (
    SegmenterConfig,
    SubtitleSegmentModel,
    EventModel,
    WordModel,
    initialize_punctuation,
)


class Segmenter:
    def __init__(self, config: SegmenterConfig):
        self.config = config
        self._punctuation = initialize_punctuation()

    def split_word_punctuation(self, word_text: str) -> List[tuple[str, str]]:
        """Split word into text and punctuation parts."""
        result = []
        starts_with = word_text[0] in self._punctuation if word_text else False
        ends_with = word_text[-1] in self._punctuation if word_text else False

        if starts_with:
            result.append((word_text[0], "punctuation"))
            word_text = word_text[1:]

        if word_text:
            if ends_with:
                if len(word_text) > 1:
                    result.append((word_text[:-1], "word"))
                result.append((word_text[-1], "punctuation"))
            else:
                result.append((word_text, "word"))

        return result

    def segment_words(
        self, words: List[WordModel], lang: Optional[str] = None
    ) -> Tuple[List[SubtitleSegmentModel], List[EventModel]]:
        """Segment words into subtitles and events."""
        segments = []
        events = []

        current_line = ""
        current_subtitle = ""
        lines = []
        if not words:
            return [], []

        segment_start = words[0].start
        segment_end = segment_start

        for i, word in enumerate(words):
            duration = word.end - segment_start
            pause = word.start - segment_end if i > 0 else 0
            segment_end = word.end
            is_last_word = i == len(words) - 1

            parts = self.split_word_punctuation(word.text)
            if not parts:
                parts = [(word.text, "word")]

            for j, (text_part, part_type) in enumerate(parts):
                event = EventModel(
                    content=text_part,
                    start_time=word.start,
                    end_time=word.end,
                    event_type=part_type,
                    confidence=word.confidence if part_type == "word" else None,
                    language=lang,
                    speaker=word.speaker if part_type == "word" else None,
                    is_eol=False,
                    is_eos=False,
                )

                if part_type == "word":
                    if current_line:
                        current_line += " "
                    current_line += text_part
                    current_subtitle += (" " if current_subtitle else "") + text_part

                is_line_break = len(current_line) >= self.config.max_chars_per_line
                is_subtitle_break = (
                    len(current_subtitle) >= self.config.max_chars_per_subtitle
                    or duration >= self.config.max_duration
                    or pause >= self.config.pause_threshold
                    or (
                        self.config.end_on_punctuation
                        and part_type == "punctuation"
                        and text_part in self.config.punctuation_marks
                    )
                    or is_last_word
                )

                # End of line
                if is_line_break:
                    event.is_eol = True
                    lines.append(current_line.strip())
                    current_line = ""

                # End of subtitle
                if is_subtitle_break:
                    if part_type == "punctuation":
                        event.is_eos = True
                    if current_line:
                        lines.append(current_line.strip())
                        current_line = ""
                    segments.append(
                        SubtitleSegmentModel(
                            start=segment_start, end=segment_end, lines=lines
                        )
                    )
                    lines = []
                    current_subtitle = ""
                    if not is_last_word:
                        segment_start = words[i + 1].start

                events.append(event)

        return segments, events

    def create_balanced_segments(
        self, words: List[WordModel], max_words_per_segment: int = 15
    ) -> List[SubtitleSegmentModel]:
        """Create balanced subtitle segments with optimal word distribution."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # Check if we should end current segment
            should_end = (
                len(current_words) >= max_words_per_segment
                or i == len(words) - 1
                or word.end - current_start > self.config.max_duration
            )

            if should_end:
                # Create segment text
                segment_text = " ".join([w.text for w in current_words])

                # Split into lines if too long
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                # Reset for next segment
                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def _split_text_into_lines(self, text: str) -> List[str]:
        """Split text into lines respecting word boundaries."""
        if len(text) <= self.config.max_chars_per_line:
            return [text]

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()

            if len(test_line) <= self.config.max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def create_speaker_aware_segments(
        self, words: List[WordModel]
    ) -> List[SubtitleSegmentModel]:
        """Create segments that respect speaker boundaries."""
        if not words:
            return []

        segments = []
        current_words = []
        current_speaker = None
        current_start = words[0].start

        for i, word in enumerate(words):
            # Check for speaker change
            if word.speaker != current_speaker and current_words:
                # End current segment
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(
                        start=current_start, end=current_words[-1].end, lines=lines
                    )
                )

                # Start new segment
                current_words = []
                current_start = word.start

            current_words.append(word)
            current_speaker = word.speaker

            # Also check other ending conditions
            should_end = (
                len(" ".join([w.text for w in current_words]))
                >= self.config.max_chars_per_subtitle
                or word.end - current_start > self.config.max_duration
                or i == len(words) - 1
            )

            if should_end:
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def create_punctuation_aware_segments(
        self, words: List[WordModel]
    ) -> List[SubtitleSegmentModel]:
        """Create segments that end on natural punctuation boundaries."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # Check if word ends with sentence-ending punctuation
            ends_with_punctuation = any(
                word.text.rstrip().endswith(punct)
                for punct in self.config.punctuation_marks
            )

            # Check ending conditions
            should_end = (
                ends_with_punctuation
                or i == len(words) - 1
                or word.end - current_start > self.config.max_duration
                or len(" ".join([w.text for w in current_words]))
                >= self.config.max_chars_per_subtitle
            )

            if should_end:
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def create_pause_aware_segments(
        self, words: List[WordModel]
    ) -> List[SubtitleSegmentModel]:
        """Create segments based on natural pauses in speech."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # Calculate pause after this word
            next_pause = 0.0
            if i < len(words) - 1:
                next_pause = words[i + 1].start - word.end

            # Check ending conditions
            should_end = (
                next_pause >= self.config.pause_threshold
                or i == len(words) - 1
                or word.end - current_start > self.config.max_duration
                or len(" ".join([w.text for w in current_words]))
                >= self.config.max_chars_per_subtitle
            )

            if should_end:
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments


class AdvancedSegmenter(Segmenter):
    """Extended segmenter with advanced features for different use cases."""

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)
        self.sentence_endings = {".", "!", "?", "。", "！", "？", "؟", "।", "؛"}
        self.clause_endings = {",", ";", ":", "，", "；", "：", "、"}

    def create_sentence_based_segments(
        self, words: List[WordModel]
    ) -> List[SubtitleSegmentModel]:
        """Create segments based on sentence boundaries."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # Check if word ends with sentence ending punctuation
            ends_sentence = any(
                word.text.rstrip().endswith(ending) for ending in self.sentence_endings
            )

            # Check if we should end segment
            should_end = (
                ends_sentence
                or i == len(words) - 1
                or word.end - current_start > self.config.max_duration
                or len(" ".join([w.text for w in current_words]))
                >= self.config.max_chars_per_subtitle
            )

            if should_end:
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def create_adaptive_segments(
        self, words: List[WordModel], target_reading_speed: float = 200
    ) -> List[SubtitleSegmentModel]:
        """Create segments based on optimal reading speed (words per minute)."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start

        # Convert reading speed to words per second
        words_per_second = target_reading_speed / 60.0

        for i, word in enumerate(words):
            current_words.append(word)
            current_duration = word.end - current_start
            current_word_count = len(current_words)

            # Calculate if current segment exceeds comfortable reading speed
            required_time = current_word_count / words_per_second
            is_too_fast = current_duration < required_time * 0.8  # Allow 20% faster

            # Check ending conditions
            should_end = (
                i == len(words) - 1
                or current_duration > self.config.max_duration
                or len(" ".join([w.text for w in current_words]))
                >= self.config.max_chars_per_subtitle
                or (
                    current_word_count >= 8 and not is_too_fast
                )  # Minimum 8 words if not too fast
            )

            if should_end:
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def create_confidence_based_segments(
        self, words: List[WordModel], min_confidence: float = 0.7
    ) -> List[SubtitleSegmentModel]:
        """Create segments considering word confidence scores."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start
        low_confidence_count = 0

        for i, word in enumerate(words):
            current_words.append(word)

            # Track low confidence words
            if word.confidence is not None and word.confidence < min_confidence:
                low_confidence_count += 1

            # Calculate average confidence for current segment
            confident_words = [
                w
                for w in current_words
                if w.confidence is not None and w.confidence >= min_confidence
            ]
            confidence_ratio = (
                len(confident_words) / len(current_words) if current_words else 1.0
            )

            # Check ending conditions
            should_end = (
                i == len(words) - 1
                or confidence_ratio < 0.6  # Too many low confidence words
                or word.end - current_start > self.config.max_duration
                or len(" ".join([w.text for w in current_words]))
                >= self.config.max_chars_per_subtitle
            )

            if should_end:
                segment_text = " ".join([w.text for w in current_words])
                lines = self._split_text_into_lines(segment_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                low_confidence_count = 0
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def create_multi_criteria_segments(
        self,
        words: List[WordModel],
        prioritize_speakers: bool = True,
        prioritize_sentences: bool = True,
        prioritize_pauses: bool = True,
        target_reading_speed: float = 200,
    ) -> List[SubtitleSegmentModel]:
        """Create segments using multiple criteria with priorities."""
        if not words:
            return []

        segments = []
        current_words = []
        current_start = words[0].start
        words_per_second = target_reading_speed / 60.0

        for i, word in enumerate(words):
            current_words.append(word)
            current_speaker = word.speaker if hasattr(word, "speaker") else None

            # Calculate various metrics
            current_duration = word.end - current_start
            current_text = " ".join([w.text for w in current_words])
            next_pause = 0.0
            if i < len(words) - 1:
                next_pause = words[i + 1].start - word.end

            # Speaker change priority
            speaker_change = False
            if prioritize_speakers and i < len(words) - 1:
                next_speaker = getattr(words[i + 1], "speaker", None)
                speaker_change = (
                    current_speaker != next_speaker and current_speaker is not None
                )

            # Sentence end priority
            sentence_end = prioritize_sentences and any(
                word.text.rstrip().endswith(ending) for ending in self.sentence_endings
            )

            # Pause priority
            long_pause = prioritize_pauses and next_pause >= self.config.pause_threshold

            # Reading speed consideration
            required_time = len(current_words) / words_per_second
            appropriate_duration = (
                abs(current_duration - required_time) < required_time * 0.3
            )

            # Decide based on priorities
            should_end = (
                i == len(words) - 1  # Last word
                or current_duration > self.config.max_duration  # Max duration exceeded
                or len(current_text)
                >= self.config.max_chars_per_subtitle  # Max chars exceeded
                or (
                    speaker_change and len(current_words) >= 5
                )  # Speaker change with minimum words
                or (
                    sentence_end and appropriate_duration and len(current_words) >= 8
                )  # Sentence end with good timing
                or (
                    long_pause and len(current_words) >= 6
                )  # Long pause with minimum words
            )

            if should_end:
                lines = self._split_text_into_lines(current_text)

                segments.append(
                    SubtitleSegmentModel(start=current_start, end=word.end, lines=lines)
                )

                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        return segments

    def analyze_segmentation_quality(
        self, segments: List[SubtitleSegmentModel], words: List[WordModel]
    ) -> dict:
        """Analyze the quality of segmentation."""
        if not segments or not words:
            return {}

        # Calculate metrics
        segment_durations = [seg.end - seg.start for seg in segments]
        segment_lengths = [len(" ".join(seg.lines)) for seg in segments]
        segment_word_counts = [len(" ".join(seg.lines).split()) for seg in segments]

        # Reading speed analysis
        reading_speeds = []
        for seg in segments:
            duration = seg.end - seg.start
            word_count = len(" ".join(seg.lines).split())
            if duration > 0:
                speed = (word_count / duration) * 60  # words per minute
                reading_speeds.append(speed)

        analysis = {
            "total_segments": len(segments),
            "avg_duration": sum(segment_durations) / len(segment_durations),
            "avg_length": sum(segment_lengths) / len(segment_lengths),
            "avg_words_per_segment": sum(segment_word_counts)
            / len(segment_word_counts),
            "avg_reading_speed": (
                sum(reading_speeds) / len(reading_speeds) if reading_speeds else 0
            ),
            "duration_distribution": {
                "min": min(segment_durations),
                "max": max(segment_durations),
                "std": self._calculate_std(segment_durations),
            },
            "length_distribution": {
                "min": min(segment_lengths),
                "max": max(segment_lengths),
                "std": self._calculate_std(segment_lengths),
            },
            "quality_score": self._calculate_quality_score(segments, words),
        }

        return analysis

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def _calculate_quality_score(
        self, segments: List[SubtitleSegmentModel], words: List[WordModel]
    ) -> float:
        """Calculate a quality score for the segmentation (0-1)."""
        if not segments or not words:
            return 0.0

        score = 1.0

        # Penalize segments that are too short or too long
        for seg in segments:
            duration = seg.end - seg.start
            if duration < 1.0:
                score -= 0.1  # Too short
            elif duration > 8.0:
                score -= 0.1  # Too long

            # Penalize segments with too few or too many characters
            char_count = len(" ".join(seg.lines))
            if char_count < 20:
                score -= 0.05  # Too short text
            elif char_count > 100:
                score -= 0.05  # Too long text

        return max(0.0, min(1.0, score))


def convert_elevenlabs_to_word_list(eleven_data: dict) -> List[WordModel]:
    """Convert ElevenLabs transcription data to a list of WordModel instances."""
    word_list = []
    for segment in eleven_data.get("segments", []):
        for word in segment.get("words", []):
            word_list.append(
                WordModel(
                    start=word["start"],
                    end=word["end"],
                    text=word["text"],
                    confidence=word.get("confidence"),
                    speaker=word.get("speaker"),
                )
            )
    return word_list


# Utility functions for segmentation strategies
def get_optimal_segmentation_strategy(
    words: List[WordModel], config: SegmenterConfig, context: dict = None
) -> str:
    """Determine the optimal segmentation strategy based on audio characteristics."""
    if not words:
        return "balanced"

    context = context or {}

    # Check if multiple speakers are present
    speakers = set(getattr(word, "speaker", None) for word in words)
    has_multiple_speakers = len([s for s in speakers if s is not None]) > 1

    # Check average confidence
    confidences = [word.confidence for word in words if word.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    # Check for sentence patterns
    sentence_endings = sum(
        1
        for word in words
        if any(
            word.text.rstrip().endswith(punct)
            for punct in [".", "!", "?", "。", "！", "？"]
        )
    )
    has_sentence_structure = sentence_endings > len(words) * 0.1

    # Decision logic
    if has_multiple_speakers:
        return "speaker_aware"
    elif avg_confidence < 0.7:
        return "confidence_based"
    elif has_sentence_structure:
        return "sentence_based"
    elif (
        context.get("content_type") == "lecture"
        or context.get("content_type") == "presentation"
    ):
        return "adaptive"
    else:
        return "multi_criteria"


def create_segments_with_strategy(
    words: List[WordModel], config: SegmenterConfig, strategy: str = "auto", **kwargs
) -> List[SubtitleSegmentModel]:
    """Create segments using the specified strategy."""
    if strategy == "auto":
        strategy = get_optimal_segmentation_strategy(words, config, kwargs)

    segmenter = AdvancedSegmenter(config)

    strategy_map = {
        "basic": segmenter.segment_words,
        "balanced": segmenter.create_balanced_segments,
        "speaker_aware": segmenter.create_speaker_aware_segments,
        "punctuation_aware": segmenter.create_punctuation_aware_segments,
        "pause_aware": segmenter.create_pause_aware_segments,
        "sentence_based": segmenter.create_sentence_based_segments,
        "adaptive": segmenter.create_adaptive_segments,
        "confidence_based": segmenter.create_confidence_based_segments,
        "multi_criteria": segmenter.create_multi_criteria_segments,
    }

    if strategy in strategy_map:
        if strategy == "basic":
            segments, _ = strategy_map[strategy](words)
            return segments
        else:
            return strategy_map[strategy](words, **kwargs)
    else:
        # Fallback to basic segmentation
        segments, _ = segmenter.segment_words(words)
        return segments
