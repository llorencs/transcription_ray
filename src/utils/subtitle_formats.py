"""
Utilities for converting transcription results to various subtitle formats.
"""

from typing import List
from datetime import timedelta
import re

from src.models.pydantic_models import SegmentModel


class SubtitleFormatter:
    """Format transcription segments into various subtitle formats."""

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    @staticmethod
    def _format_time_vtt(seconds: float) -> str:
        """Format seconds to VTT time format (HH:MM:SS.mmm)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for subtitle display."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove or replace problematic characters
        text = text.replace("\n", " ").replace("\r", " ")

        return text

    def to_srt(self, segments: List[SegmentModel]) -> str:
        """Convert segments to SRT format."""
        srt_content = []

        for i, segment in enumerate(segments, 1):
            start_time = self._format_time_srt(segment.start)
            end_time = self._format_time_srt(segment.end)
            text = self._clean_text(segment.text)

            # Add speaker information if available
            if hasattr(segment, "speaker") and segment.speaker:
                text = f"[{segment.speaker}] {text}"

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # Empty line between subtitles

        return "\n".join(srt_content)

    def to_vtt(self, segments: List[SegmentModel]) -> str:
        """Convert segments to WebVTT format."""
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            start_time = self._format_time_vtt(segment.start)
            end_time = self._format_time_vtt(segment.end)
            text = self._clean_text(segment.text)

            # Add speaker information if available
            if hasattr(segment, "speaker") and segment.speaker:
                text = f"<v {segment.speaker}>{text}</v>"

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")  # Empty line between subtitles

        return "\n".join(vtt_content)

    def to_ass(self, segments: List[SegmentModel]) -> str:
        """Convert segments to ASS (Advanced SubStation Alpha) format."""
        ass_header = """[Script Info]
Title: Transcription
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        ass_content = [ass_header]

        for segment in segments:
            start_time = self._format_time_ass(segment.start)
            end_time = self._format_time_ass(segment.end)
            text = self._clean_text(segment.text)

            # Add speaker information if available
            speaker_name = getattr(segment, "speaker", "") or ""

            ass_content.append(
                f"Dialogue: 0,{start_time},{end_time},Default,{speaker_name},0,0,0,,{text}"
            )

        return "\n".join(ass_content)

    @staticmethod
    def _format_time_ass(seconds: float) -> str:
        """Format seconds to ASS time format (H:MM:SS.cc)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        centiseconds = int((seconds - int(seconds)) * 100)

        return f"{hours}:{minutes:02d}:{int(seconds):02d}.{centiseconds:02d}"

    def to_ttml(self, segments: List[SegmentModel]) -> str:
        """Convert segments to TTML (Timed Text Markup Language) format."""
        ttml_content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">',
            "<head>",
            "<styling>",
            '<style xml:id="defaultStyle" tts:fontFamily="Arial" tts:fontSize="16px" tts:color="white" tts:textAlign="center"/>',
            "</styling>",
            "</head>",
            "<body>",
            "<div>",
        ]

        for segment in segments:
            start_time = self._format_time_ttml(segment.start)
            end_time = self._format_time_ttml(segment.end)
            text = self._clean_text(segment.text)

            # Escape XML characters
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            # Add speaker information if available
            if hasattr(segment, "speaker") and segment.speaker:
                text = f"[{segment.speaker}] {text}"

            ttml_content.append(
                f'<p begin="{start_time}" end="{end_time}" style="defaultStyle">{text}</p>'
            )

        ttml_content.extend(["</div>", "</body>", "</tt>"])

        return "\n".join(ttml_content)

    @staticmethod
    def _format_time_ttml(seconds: float) -> str:
        """Format seconds to TTML time format (HH:MM:SS.mmm)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

    def to_json_segments(self, segments: List[SegmentModel]) -> List[dict]:
        """Convert segments to JSON format with additional metadata."""
        json_segments = []

        for i, segment in enumerate(segments):
            segment_dict = {
                "index": i,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.end - segment.start,
                "text": self._clean_text(segment.text),
                "word_count": len(segment.text.split()),
                "words": [],
            }

            # Add speaker information if available
            if hasattr(segment, "speaker") and segment.speaker:
                segment_dict["speaker"] = segment.speaker

            # Add word-level information if available
            if hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    word_dict = {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        "duration": word.end - word.start,
                    }

                    if hasattr(word, "confidence") and word.confidence is not None:
                        word_dict["confidence"] = word.confidence

                    if hasattr(word, "speaker") and word.speaker:
                        word_dict["speaker"] = word.speaker

                    segment_dict["words"].append(word_dict)

            json_segments.append(segment_dict)

        return json_segments

    def to_csv(self, segments: List[SegmentModel]) -> str:
        """Convert segments to CSV format."""
        csv_lines = ["Index,Start,End,Duration,Text,Speaker,Word_Count"]

        for i, segment in enumerate(segments):
            text = self._clean_text(segment.text).replace('"', '""')  # Escape quotes
            speaker = getattr(segment, "speaker", "") or ""
            duration = segment.end - segment.start
            word_count = len(segment.text.split())

            csv_lines.append(
                f'{i},"{segment.start:.3f}","{segment.end:.3f}","{duration:.3f}","{text}","{speaker}",{word_count}'
            )

        return "\n".join(csv_lines)

    def to_transcript(
        self,
        segments: List[SegmentModel],
        include_timestamps: bool = True,
        include_speakers: bool = True,
    ) -> str:
        """Convert segments to readable transcript format."""
        transcript_lines = []
        current_speaker = None

        for segment in segments:
            text = self._clean_text(segment.text)
            speaker = getattr(segment, "speaker", None)

            # Add timestamp if requested
            if include_timestamps:
                timestamp = f"[{segment.start:.1f}s]"

                # Add speaker change if needed
                if include_speakers and speaker and speaker != current_speaker:
                    transcript_lines.append(f"\n{speaker}:")
                    current_speaker = speaker

                transcript_lines.append(f"{timestamp} {text}")
            else:
                # Add speaker change if needed
                if include_speakers and speaker and speaker != current_speaker:
                    transcript_lines.append(f"\n{speaker}: {text}")
                    current_speaker = speaker
                else:
                    transcript_lines.append(text)

        return "\n".join(transcript_lines)

    def create_chapters(
        self, segments: List[SegmentModel], min_chapter_duration: float = 60.0
    ) -> List[dict]:
        """Create chapter markers based on speaker changes or time intervals."""
        chapters = []
        current_chapter_start = 0.0
        current_speaker = None
        chapter_text = []

        for segment in segments:
            speaker = getattr(segment, "speaker", None)

            # Create new chapter on speaker change or time threshold
            if (speaker and speaker != current_speaker) or (
                segment.start - current_chapter_start >= min_chapter_duration
            ):

                if chapter_text:
                    chapters.append(
                        {
                            "index": len(chapters),
                            "start": current_chapter_start,
                            "end": segment.start,
                            "duration": segment.start - current_chapter_start,
                            "speaker": current_speaker or "Unknown",
                            "title": f"Chapter {len(chapters) + 1}"
                            + (f" - {current_speaker}" if current_speaker else ""),
                            "text": " ".join(chapter_text),
                        }
                    )

                current_chapter_start = segment.start
                current_speaker = speaker
                chapter_text = []

            chapter_text.append(self._clean_text(segment.text))

        # Add final chapter
        if chapter_text and segments:
            chapters.append(
                {
                    "index": len(chapters),
                    "start": current_chapter_start,
                    "end": segments[-1].end,
                    "duration": segments[-1].end - current_chapter_start,
                    "speaker": current_speaker or "Unknown",
                    "title": f"Chapter {len(chapters) + 1}"
                    + (f" - {current_speaker}" if current_speaker else ""),
                    "text": " ".join(chapter_text),
                }
            )

        return chapters
