"""
src/services/preprocessing_actors.py

Preprocessing actors for audio enhancement using Demucs and noise reduction.
"""

import os
import ray
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any, List
import noisereduce as nr
from demucs.pretrained import get_model
from demucs.apply import apply_model


@ray.remote
class DemucsActor:
    """Ray actor for music/vocals separation using Demucs."""

    def __init__(self, model_name: str = "htdemucs", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Demucs model."""
        try:
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # Load model
            self.model = get_model(self.model_name)
            self.model.to(device)
            self.model.eval()

            print(f"Loaded Demucs model {self.model_name} on {device}")

        except Exception as e:
            print(f"Failed to load Demucs model: {e}")
            raise

    def separate_vocals(self, audio_path: str, output_dir: Optional[str] = None) -> str:
        """Separate vocals from music and return vocals-only audio path."""
        try:
            # Create output directory if not provided
            if output_dir is None:
                output_dir = Path(tempfile.gettempdir()) / "demucs_output"
                output_dir.mkdir(exist_ok=True)

            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.model.samplerate, mono=False)

            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            # Move to device
            device = next(self.model.parameters()).device
            audio_tensor = audio_tensor.to(device)

            # Separate sources
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor)

            # Extract vocals (usually the last source)
            vocals = sources[0, -1].cpu().numpy()  # [channels, samples]

            # Save vocals
            vocals_path = Path(output_dir) / f"vocals_{Path(audio_path).stem}.wav"
            sf.write(vocals_path, vocals.T, sr)  # Transpose for soundfile

            return str(vocals_path)

        except Exception as e:
            print(f"Vocal separation failed: {e}")
            raise

    def enhance_vocals(self, audio_path: str, vocal_boost: float = 1.2) -> str:
        """Enhance vocals in audio by separating and boosting."""
        try:
            # Separate vocals and accompaniment
            output_dir = Path(tempfile.gettempdir()) / "demucs_enhance"
            output_dir.mkdir(exist_ok=True)

            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.model.samplerate, mono=False)

            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            # Move to device
            device = next(self.model.parameters()).device
            audio_tensor = audio_tensor.to(device)

            # Separate sources
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor)

            # Get vocals and accompaniment
            vocals = sources[0, -1].cpu().numpy()  # vocals
            accompaniment = (
                sources[0, :-1].sum(dim=0).cpu().numpy()
            )  # sum other sources

            # Boost vocals and recombine
            enhanced = (
                vocals * vocal_boost + accompaniment * 0.3
            )  # Reduce accompaniment

            # Normalize to prevent clipping
            max_val = np.abs(enhanced).max()
            if max_val > 0.95:
                enhanced = enhanced * 0.95 / max_val

            # Save enhanced audio
            enhanced_path = Path(output_dir) / f"enhanced_{Path(audio_path).stem}.wav"
            sf.write(enhanced_path, enhanced.T, sr)

            return str(enhanced_path)

        except Exception as e:
            print(f"Vocal enhancement failed: {e}")
            raise


@ray.remote
class NoiseReductionActor:
    """Ray actor for noise reduction."""

    def __init__(self):
        pass

    def reduce_noise(
        self, audio_path: str, stationary: bool = True, prop_decrease: float = 1.0
    ) -> str:
        """Reduce noise in audio file."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)

            # Apply noise reduction
            if stationary:
                # For stationary noise (constant background noise)
                reduced_audio = nr.reduce_noise(
                    y=audio, sr=sr, stationary=True, prop_decrease=prop_decrease
                )
            else:
                # For non-stationary noise
                reduced_audio = nr.reduce_noise(
                    y=audio, sr=sr, stationary=False, prop_decrease=prop_decrease
                )

            # Save processed audio
            output_path = (
                Path(tempfile.gettempdir()) / f"denoised_{Path(audio_path).stem}.wav"
            )
            sf.write(output_path, reduced_audio, sr)

            return str(output_path)

        except Exception as e:
            print(f"Noise reduction failed: {e}")
            raise

    def reduce_noise_spectral(
        self, audio_path: str, noise_gate_threshold: float = -40
    ) -> str:
        """Apply spectral noise reduction with noise gate."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)

            # Apply spectral subtraction-based noise reduction
            # Convert to dB
            audio_db = librosa.amplitude_to_db(np.abs(audio))

            # Apply noise gate (silence parts below threshold)
            mask = audio_db > noise_gate_threshold
            audio_gated = audio * mask

            # Apply spectral smoothing
            stft = librosa.stft(audio_gated)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Smooth magnitude spectrum
            magnitude_smoothed = librosa.filters.median_filter(
                magnitude, width=3, axis=0
            )

            # Reconstruct audio
            stft_reconstructed = magnitude_smoothed * np.exp(1j * phase)
            audio_processed = librosa.istft(stft_reconstructed)

            # Save processed audio
            output_path = (
                Path(tempfile.gettempdir())
                / f"spectral_denoised_{Path(audio_path).stem}.wav"
            )
            sf.write(output_path, audio_processed, sr)

            return str(output_path)

        except Exception as e:
            print(f"Spectral noise reduction failed: {e}")
            raise


@ray.remote
class AudioNormalizationActor:
    """Ray actor for audio normalization and enhancement."""

    def __init__(self):
        pass

    def normalize_audio(self, audio_path: str, target_lufs: float = -23.0) -> str:
        """Normalize audio to target LUFS using ffmpeg."""
        try:
            output_path = (
                Path(tempfile.gettempdir()) / f"normalized_{Path(audio_path).stem}.wav"
            )

            # Use ffmpeg for professional audio normalization
            cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-af",
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-ar",
                "16000",  # Standard sample rate for speech recognition
                "-ac",
                "1",  # Convert to mono
                "-y",  # Overwrite output
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"FFmpeg normalization failed: {result.stderr}")

            return str(output_path)

        except Exception as e:
            print(f"Audio normalization failed: {e}")
            raise

    def enhance_speech(self, audio_path: str) -> str:
        """Enhance speech audio for better transcription."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Apply speech enhancement filters
            # 1. High-pass filter to remove low-frequency noise
            audio_filtered = librosa.effects.preemphasis(audio, coef=0.97)

            # 2. Apply mild compression
            # Simple dynamic range compression
            threshold = 0.1
            ratio = 3.0
            above_threshold = np.abs(audio_filtered) > threshold
            compressed = np.copy(audio_filtered)
            compressed[above_threshold] = np.sign(audio_filtered[above_threshold]) * (
                threshold
                + (np.abs(audio_filtered[above_threshold]) - threshold) / ratio
            )

            # 3. Normalize
            if np.max(np.abs(compressed)) > 0:
                compressed = compressed / np.max(np.abs(compressed)) * 0.8

            # Save enhanced audio
            output_path = (
                Path(tempfile.gettempdir())
                / f"speech_enhanced_{Path(audio_path).stem}.wav"
            )
            sf.write(output_path, compressed, sr)

            return str(output_path)

        except Exception as e:
            print(f"Speech enhancement failed: {e}")
            raise


@ray.remote
class PreprocessingActor:
    """Main preprocessing actor that coordinates all preprocessing steps."""

    def __init__(self):
        self.demucs_actor = None
        self.noise_actor = None
        self.normalize_actor = None

    def _initialize_actors(self):
        """Initialize preprocessing actors."""
        if not self.demucs_actor:
            self.demucs_actor = DemucsActor.remote()
        if not self.noise_actor:
            self.noise_actor = NoiseReductionActor.remote()
        if not self.normalize_actor:
            self.normalize_actor = AudioNormalizationActor.remote()

    async def preprocess_audio(
        self,
        audio_path: str,
        separate_vocals: bool = True,
        reduce_noise: bool = True,
        normalize: bool = True,
        enhance_speech: bool = True,
    ) -> str:
        """Complete audio preprocessing pipeline."""
        try:
            self._initialize_actors()

            current_path = audio_path
            temp_files = []

            print(f"Starting preprocessing of {audio_path}")

            # Step 1: Vocal separation (if requested and music is detected)
            if separate_vocals:
                try:
                    print("Separating vocals from music...")
                    vocals_path = await self.demucs_actor.separate_vocals.remote(
                        current_path
                    )
                    temp_files.append(vocals_path)
                    current_path = vocals_path
                    print("Vocal separation completed")
                except Exception as e:
                    print(f"Vocal separation failed, continuing with original: {e}")

            # Step 2: Noise reduction
            if reduce_noise:
                try:
                    print("Reducing background noise...")
                    denoised_path = await self.noise_actor.reduce_noise.remote(
                        current_path, stationary=True, prop_decrease=0.8
                    )
                    temp_files.append(denoised_path)
                    current_path = denoised_path
                    print("Noise reduction completed")
                except Exception as e:
                    print(f"Noise reduction failed, continuing: {e}")

            # Step 3: Speech enhancement
            if enhance_speech:
                try:
                    print("Enhancing speech...")
                    enhanced_path = await self.normalize_actor.enhance_speech.remote(
                        current_path
                    )
                    temp_files.append(enhanced_path)
                    current_path = enhanced_path
                    print("Speech enhancement completed")
                except Exception as e:
                    print(f"Speech enhancement failed, continuing: {e}")

            # Step 4: Final normalization
            if normalize:
                try:
                    print("Normalizing audio...")
                    normalized_path = await self.normalize_actor.normalize_audio.remote(
                        current_path, target_lufs=-23.0
                    )
                    temp_files.append(normalized_path)
                    current_path = normalized_path
                    print("Audio normalization completed")
                except Exception as e:
                    print(f"Audio normalization failed, continuing: {e}")

            # Create final output file
            final_output = (
                Path(tempfile.gettempdir())
                / f"preprocessed_{Path(audio_path).stem}.wav"
            )

            if current_path != audio_path:
                # Copy final result
                import shutil

                shutil.copy2(current_path, final_output)

                # Cleanup temporary files except the final one
                for temp_file in temp_files:
                    try:
                        if temp_file != current_path:
                            Path(temp_file).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"Failed to cleanup temp file {temp_file}: {e}")
            else:
                # No preprocessing was successful, return original
                final_output = Path(audio_path)

            print(f"Preprocessing completed: {final_output}")
            return str(final_output)

        except Exception as e:
            print(f"Preprocessing pipeline failed: {e}")
            # Return original file if preprocessing fails
            return audio_path

    async def preprocess_for_transcription(self, audio_path: str) -> str:
        """Optimized preprocessing specifically for transcription accuracy."""
        try:
            self._initialize_actors()

            print(f"Preprocessing for transcription: {audio_path}")

            # Load audio to analyze characteristics
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr

            print(f"Audio duration: {duration:.2f}s, Sample rate: {sr}Hz")

            current_path = audio_path

            # Step 1: Check if vocal separation is needed
            # Only separate if we detect significant non-vocal content
            rms_energy = librosa.feature.rms(y=audio)[0]
            energy_variance = np.var(rms_energy)

            if energy_variance > 0.01:  # High variance suggests music/mixed content
                try:
                    print("Detected mixed content, enhancing vocals...")
                    enhanced_path = await self.demucs_actor.enhance_vocals.remote(
                        current_path, vocal_boost=1.3
                    )
                    current_path = enhanced_path
                except Exception as e:
                    print(f"Vocal enhancement failed: {e}")

            # Step 2: Targeted noise reduction
            try:
                print("Applying targeted noise reduction...")
                denoised_path = await self.noise_actor.reduce_noise_spectral.remote(
                    current_path, noise_gate_threshold=-35
                )
                current_path = denoised_path
            except Exception as e:
                print(f"Spectral noise reduction failed: {e}")

            # Step 3: Speech optimization
            try:
                print("Optimizing for speech recognition...")
                optimized_path = await self.normalize_actor.enhance_speech.remote(
                    current_path
                )
                current_path = optimized_path
            except Exception as e:
                print(f"Speech optimization failed: {e}")

            # Step 4: Final normalization for ASR
            try:
                print("Final normalization...")
                final_path = await self.normalize_actor.normalize_audio.remote(
                    current_path, target_lufs=-20.0  # Slightly higher for ASR
                )
                current_path = final_path
            except Exception as e:
                print(f"Final normalization failed: {e}")

            print(f"Transcription preprocessing completed: {current_path}")
            return current_path

        except Exception as e:
            print(f"Transcription preprocessing failed: {e}")
            return audio_path

    async def analyze_audio_content(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio to determine optimal preprocessing strategy."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)

            # Basic analysis
            duration = len(audio) / sr
            rms_energy = librosa.feature.rms(y=audio)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

            # Calculate statistics
            analysis = {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if audio.ndim == 1 else audio.shape[0],
                "rms_mean": float(np.mean(rms_energy)),
                "rms_std": float(np.std(rms_energy)),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "zcr_mean": float(np.mean(zero_crossing_rate)),
                "dynamic_range": float(np.max(rms_energy) - np.min(rms_energy)),
                "recommendations": {},
            }

            # Generate recommendations based on analysis
            if analysis["rms_std"] > 0.02:
                analysis["recommendations"]["vocal_separation"] = "recommended"
            else:
                analysis["recommendations"]["vocal_separation"] = "not_needed"

            if analysis["zcr_mean"] > 0.1:
                analysis["recommendations"]["noise_reduction"] = "aggressive"
            elif analysis["zcr_mean"] > 0.05:
                analysis["recommendations"]["noise_reduction"] = "moderate"
            else:
                analysis["recommendations"]["noise_reduction"] = "light"

            if analysis["spectral_centroid_mean"] < 2000:
                analysis["recommendations"][
                    "speech_enhancement"
                ] = "high_frequency_boost"
            else:
                analysis["recommendations"]["speech_enhancement"] = "standard"

            return analysis

        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return {
                "error": str(e),
                "recommendations": {
                    "vocal_separation": "skip",
                    "noise_reduction": "light",
                    "speech_enhancement": "standard",
                },
            }
