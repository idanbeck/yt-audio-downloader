#!/usr/bin/env python3
"""YouTube Audio Downloader - Downloads audio from YouTube URLs in a YAML file."""

import re
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

# Intensity labels (1-10 scale)
INTENSITY_LABELS = {
    1: "01",
    2: "02", 
    3: "03",
    4: "04",
    5: "05",
    6: "06",
    7: "07",
    8: "08",
    9: "09",
    10: "10",
}

# BPM curve options for mix creation
BPM_CURVES = {
    "up": "Start slow, build energy throughout",
    "down": "Start high energy, wind down",
    "peak": "Build to peak in middle, then wind down",
    "valley": "Start/end high, dip in middle",
    "smooth": "Minimize BPM jumps between songs",
    "none": "No BPM-based ordering (random/as specified)",
}

# Mix transition profiles
TRANSITION_PROFILES = {
    "dj": {
        "description": "Quick DJ-style cuts on beat boundaries",
        "crossfade_ms": 500,
        "fade_type": "equal_power",
        "cut_on_beat": True,
        "overlap_beats": 4,
    },
    "smooth": {
        "description": "Long gradual crossfades",
        "crossfade_ms": 6000,
        "fade_type": "linear",
        "cut_on_beat": False,
        "overlap_beats": 16,
    },
    "radio": {
        "description": "Medium crossfades, radio style",
        "crossfade_ms": 3000,
        "fade_type": "exponential",
        "cut_on_beat": False,
        "overlap_beats": 8,
    },
    "drop": {
        "description": "Cut during build-up, drop into chorus",
        "crossfade_ms": 100,
        "fade_type": "hard_cut",
        "cut_on_beat": True,
        "overlap_beats": 1,
    },
}

# Mastering profiles
MASTER_PROFILES = {
    "commercial": {
        "description": "Polished, punchy, radio-ready (preserves dynamics)",
        "target_lufs": -11.0,
        "ceiling_db": -0.5,
        "bass_boost_db": 1.5,
        "presence_boost_db": 1.0,
        "stereo_width": 1.15,
        "compression_threshold": -20,
        "compression_ratio": 2.5,
        "attack_ms": 15,
        "release_ms": 150,
        "noise_gate": True,
        "gate_threshold_db": -50,
        "de_ess": True,
        "de_ess_freq": 6000,
        "de_ess_threshold": -20,
    },
    "streaming": {
        "description": "Optimized for Spotify/Apple Music (-14 LUFS)",
        "target_lufs": -14.0,
        "ceiling_db": -1.0,
        "bass_boost_db": 1.0,
        "presence_boost_db": 0.5,
        "stereo_width": 1.1,
        "compression_threshold": -22,
        "compression_ratio": 2.0,
        "attack_ms": 20,
        "release_ms": 200,
        "noise_gate": True,
        "gate_threshold_db": -55,
        "de_ess": True,
        "de_ess_freq": 6000,
        "de_ess_threshold": -18,
    },
    "vinyl": {
        "description": "Warm, dynamic, vintage feel",
        "target_lufs": -12.0,
        "ceiling_db": -1.0,
        "bass_boost_db": 1.5,
        "presence_boost_db": 0.5,
        "stereo_width": 1.0,
        "compression_threshold": -24,
        "compression_ratio": 1.8,
        "attack_ms": 25,
        "release_ms": 250,
        "noise_gate": False,
        "de_ess": False,
        "add_warmth": True,
    },
    "dynamic": {
        "description": "Maximum dynamic range, minimal processing",
        "target_lufs": -16.0,
        "ceiling_db": -1.5,
        "bass_boost_db": 0.5,
        "presence_boost_db": 0.5,
        "stereo_width": 1.0,
        "compression_threshold": -30,
        "compression_ratio": 1.5,
        "attack_ms": 30,
        "release_ms": 300,
        "noise_gate": True,
        "gate_threshold_db": -55,
        "de_ess": False,
    },
}


@dataclass
class SongAnalysis:
    """Analyzed metadata for a song."""
    file_path: Path
    name: str
    artist: str
    duration_sec: float
    bpm: float
    beat_frames: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_times: np.ndarray = field(default_factory=lambda: np.array([]))
    downbeats: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    energy_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    intensity: int = 5  # Now 1-10 scale
    best_start_sec: float = 0.0
    best_end_sec: float = 0.0
    good_transition_in: list = field(default_factory=list)
    good_transition_out: list = field(default_factory=list)
    yaml_bpm: float = None
    yaml_timestamps: dict = field(default_factory=dict)
    yaml_best_sections: list = field(default_factory=list)


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness in LUFS."""
    import pyloudnorm as pyln
    meter = pyln.Meter(sample_rate)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return meter.integrated_loudness(audio)


def detect_recording_era(audio: np.ndarray, sample_rate: int) -> dict:
    """Analyze audio to detect characteristics of older recordings."""
    import librosa
    y_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sample_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_mono, sr=sample_rate))
    rms = librosa.feature.rms(y=y_mono)[0]
    quiet_threshold = np.percentile(rms, 10)
    quiet_mask = rms < quiet_threshold * 1.5
    noise_floor_db = -60
    if np.any(quiet_mask):
        quiet_frames = np.where(quiet_mask)[0]
        if len(quiet_frames) > 10:
            frame_length, hop_length = 2048, 512
            noise_samples = []
            for frame_idx in quiet_frames[:20]:
                start, end = frame_idx * hop_length, frame_idx * hop_length + frame_length
                if end < len(y_mono):
                    noise_samples.append(np.sqrt(np.mean(y_mono[start:end] ** 2)))
            if noise_samples:
                noise_rms = np.median(noise_samples)
                if noise_rms > 0:
                    noise_floor_db = 20 * np.log10(noise_rms + 1e-10)
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sample_rate)
    mid_high_mask = (freqs >= 2000) & (freqs <= 6000)
    high_mask = (freqs >= 6000) & (freqs <= 12000)
    mid_high_energy = np.mean(S[mid_high_mask, :])
    high_energy = np.mean(S[high_mask, :])
    harsh_ratio = high_energy / (mid_high_energy + 1e-10)
    is_harsh = harsh_ratio > 0.4
    if audio.ndim > 1 and audio.shape[1] == 2:
        try:
            correlation = np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]
            is_narrow_stereo = correlation > 0.85 if not np.isnan(correlation) else True
        except:
            is_narrow_stereo = True
    else:
        is_narrow_stereo = True
    era = "modern"
    if noise_floor_db > -50:
        era = "vintage"
    elif noise_floor_db > -55 and is_harsh:
        era = "80s_digital"
    elif is_narrow_stereo and spectral_rolloff < 8000:
        era = "70s_analog"
    return {"era": era, "noise_floor_db": noise_floor_db, "is_harsh": is_harsh,
            "harsh_ratio": harsh_ratio, "is_narrow_stereo": is_narrow_stereo,
            "spectral_centroid": spectral_centroid, "needs_noise_reduction": noise_floor_db > -50,
            "needs_de_harsh": is_harsh}


def detect_if_already_mastered(audio: np.ndarray, sample_rate: int, target_profile: dict) -> dict:
    """Detect if a track is already professionally mastered."""
    import librosa
    y_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    try:
        current_lufs = measure_lufs(audio, sample_rate)
    except:
        current_lufs = -14.0
    target_lufs = target_profile.get("target_lufs", -14.0)
    lufs_diff = abs(current_lufs - target_lufs)
    lufs_ok = lufs_diff < 3.0
    rms = librosa.feature.rms(y=y_mono)[0]
    peak = np.max(np.abs(y_mono))
    avg_rms = np.mean(rms)
    crest_factor_db = 20 * np.log10(peak / avg_rms) if avg_rms > 0 else 0
    dynamics_ok = 7 <= crest_factor_db <= 18
    is_over_compressed = crest_factor_db < 6
    clip_threshold = 0.99
    clipped_samples = np.sum(np.abs(audio) > clip_threshold)
    has_clipping = (clipped_samples / audio.size) > 0.001
    quiet_threshold = np.percentile(rms, 10)
    quiet_mask = rms < quiet_threshold * 1.5
    noise_floor_db = -70
    if np.any(quiet_mask):
        quiet_frames = np.where(quiet_mask)[0]
        if len(quiet_frames) > 5:
            frame_length, hop_length = 2048, 512
            noise_samples = []
            for frame_idx in quiet_frames[:15]:
                start, end = frame_idx * hop_length, frame_idx * hop_length + frame_length
                if end < len(y_mono):
                    noise_samples.append(np.sqrt(np.mean(y_mono[start:end] ** 2)))
            if noise_samples:
                noise_rms = np.median(noise_samples)
                if noise_rms > 0:
                    noise_floor_db = 20 * np.log10(noise_rms + 1e-10)
    noise_ok = noise_floor_db < -55
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sample_rate)
    low_mask = (freqs >= 60) & (freqs <= 250)
    mid_mask = (freqs >= 250) & (freqs <= 4000)
    low_energy = np.mean(S[low_mask, :]) if np.any(low_mask) else 0
    mid_energy = np.mean(S[mid_mask, :]) if np.any(mid_mask) else 0
    high_energy = np.mean(S[(freqs >= 4000) & (freqs <= 16000), :])
    total_energy = low_energy + mid_energy + high_energy + 1e-10
    low_ratio, mid_ratio = low_energy / total_energy, mid_energy / total_energy
    freq_balance_ok = (0.15 <= low_ratio <= 0.45) and (0.35 <= mid_ratio <= 0.65)
    is_already_mastered = lufs_ok and dynamics_ok and not has_clipping and noise_ok and not is_over_compressed
    needs_work_score = 0
    if not lufs_ok: needs_work_score += lufs_diff / 3
    if not dynamics_ok: needs_work_score += 2
    if is_over_compressed: needs_work_score += 1
    if has_clipping: needs_work_score += 2
    if not noise_ok: needs_work_score += 1.5
    if not freq_balance_ok: needs_work_score += 1
    return {"is_already_mastered": is_already_mastered, "needs_work_score": needs_work_score,
            "current_lufs": current_lufs, "target_lufs": target_lufs, "lufs_diff": lufs_diff,
            "lufs_ok": lufs_ok, "crest_factor_db": crest_factor_db, "dynamics_ok": dynamics_ok,
            "is_over_compressed": is_over_compressed, "has_clipping": has_clipping,
            "noise_floor_db": noise_floor_db, "noise_ok": noise_ok, "freq_balance_ok": freq_balance_ok,
            "low_ratio": low_ratio, "mid_ratio": mid_ratio, "high_ratio": high_energy/total_energy}


def apply_noise_reduction(audio: np.ndarray, sample_rate: int, strength: float = 0.5) -> np.ndarray:
    """Apply spectral noise reduction for older recordings."""
    from scipy import signal
    for ch in range(audio.shape[1] if audio.ndim > 1 else 1):
        channel = audio[:, ch] if audio.ndim > 1 else audio
        f, t, Zxx = signal.stft(channel, sample_rate, nperseg=2048)
        frame_power = np.mean(np.abs(Zxx) ** 2, axis=0)
        noise_threshold = np.percentile(frame_power, 10)
        noise_frames = frame_power < noise_threshold * 2
        if np.any(noise_frames):
            noise_spectrum = np.mean(np.abs(Zxx[:, noise_frames]), axis=1)
            magnitude, phase = np.abs(Zxx), np.angle(Zxx)
            noise_estimate = noise_spectrum[:, np.newaxis] * strength * 2
            magnitude_cleaned = np.maximum(magnitude - noise_estimate, magnitude * 0.1)
            Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase)
            _, channel_cleaned = signal.istft(Zxx_cleaned, sample_rate, nperseg=2048)
            if len(channel_cleaned) > len(channel):
                channel_cleaned = channel_cleaned[:len(channel)]
            elif len(channel_cleaned) < len(channel):
                channel_cleaned = np.pad(channel_cleaned, (0, len(channel) - len(channel_cleaned)))
            if audio.ndim > 1:
                audio[:, ch] = channel_cleaned
            else:
                audio = channel_cleaned
    return audio


def apply_de_harsh(audio: np.ndarray, sample_rate: int, harsh_freq: float = 4000) -> np.ndarray:
    """Reduce harsh high frequencies common in 80s recordings."""
    from scipy import signal
    nyquist = sample_rate / 2
    if harsh_freq >= nyquist:
        return audio
    for ch in range(audio.shape[1] if audio.ndim > 1 else 1):
        channel = audio[:, ch] if audio.ndim > 1 else audio
        sos_low = signal.butter(4, harsh_freq / nyquist, btype='low', output='sos')
        sos_high = signal.butter(4, harsh_freq / nyquist, btype='high', output='sos')
        low_band = signal.sosfilt(sos_low, channel)
        high_band = signal.sosfilt(sos_high, channel)
        threshold = np.percentile(np.abs(high_band), 70)
        envelope = np.abs(high_band)
        window = int(sample_rate * 0.01)
        envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
        gain = np.ones_like(envelope)
        above = envelope > threshold
        gain[above] = threshold / envelope[above]
        gain[above] = threshold + (envelope[above] - threshold) / 3.0
        gain[above] = gain[above] / envelope[above]
        gain = np.convolve(gain, np.ones(window)/window, mode='same')
        gain = np.clip(gain, 0.3, 1.0)
        result = low_band + high_band * gain * 0.85
        if audio.ndim > 1:
            audio[:, ch] = result
        else:
            audio = result
    return audio


def master_audio(file_path: Path, profile_name: str = "commercial", verbose: bool = False, force: bool = False) -> tuple[Path | None, str]:
    """Apply mastering chain to audio file."""
    from pedalboard import (Pedalboard, Compressor, Gain, Limiter, NoiseGate,
                            HighpassFilter, LowShelfFilter, HighShelfFilter, PeakFilter)
    profile = MASTER_PROFILES[profile_name]
    audio, sample_rate = sf.read(str(file_path))
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    original_lufs = measure_lufs(audio, sample_rate)
    if verbose:
        console.print(f"    [dim]Original LUFS: {original_lufs:.1f}[/dim]")
    mastered_check = detect_if_already_mastered(audio, sample_rate, profile)
    if verbose:
        console.print(f"    [dim]Quality score: {mastered_check['needs_work_score']:.1f}[/dim]")
    if mastered_check["is_already_mastered"] and not force:
        if verbose:
            console.print(f"    [green]Track already well-mastered, skipping[/green]")
        stem = file_path.stem
        output_path = file_path.parent / f"{stem}_mastered.mp3"
        if mastered_check["lufs_diff"] > 1.0:
            gain_db = mastered_check["target_lufs"] - mastered_check["current_lufs"]
            audio = audio * (10 ** (gain_db / 20))
            audio = np.clip(audio, -1.0, 1.0)
            temp_wav = file_path.parent / f"{stem}_temp.wav"
            sf.write(str(temp_wav), audio, sample_rate)
            subprocess.run(["ffmpeg", "-y", "-i", str(temp_wav), "-codec:a", "libmp3lame", "-b:a", "320k", str(output_path)], capture_output=True)
            temp_wav.unlink()
            return output_path, "light_touch"
        else:
            shutil.copy(file_path, output_path)
            return output_path, "skipped_already_good"
    era_info = detect_recording_era(audio, sample_rate)
    if verbose:
        console.print(f"    [dim]Detected era: {era_info['era']}[/dim]")
    light_processing = mastered_check["needs_work_score"] < 2.0
    # Preprocessing
    if (era_info["needs_noise_reduction"] or era_info["era"] in ["vintage", "70s_analog"]) and not (light_processing and mastered_check["noise_ok"]):
        audio = apply_noise_reduction(audio, sample_rate, strength=0.6)
    if (era_info["needs_de_harsh"] or era_info["era"] == "80s_digital") and not light_processing:
        audio = apply_de_harsh(audio, sample_rate, harsh_freq=4500)
    # Cleanup
    cleanup_chain = [HighpassFilter(cutoff_frequency_hz=35)]
    if profile.get("noise_gate", False):
        gate_threshold = profile.get("gate_threshold_db", -50)
        if era_info["era"] in ["vintage", "70s_analog"]:
            gate_threshold = min(gate_threshold, era_info["noise_floor_db"] + 5)
        cleanup_chain.append(NoiseGate(threshold_db=gate_threshold, ratio=2.0, attack_ms=1.0, release_ms=100.0))
    audio = Pedalboard(cleanup_chain)(audio, sample_rate)
    # De-essing
    if profile.get("de_ess", False):
        from scipy import signal
        de_ess_freq = profile.get("de_ess_freq", 6000)
        nyquist = sample_rate / 2
        if de_ess_freq < nyquist:
            y_mono = np.mean(audio, axis=1)
            sos = signal.butter(4, de_ess_freq / nyquist, btype='high', output='sos')
            hf_content = signal.sosfilt(sos, y_mono)
            envelope = np.abs(hf_content)
            window_size = int(sample_rate * 0.01)
            envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            threshold_linear = 10 ** (profile.get("de_ess_threshold", -20) / 20)
            gain_reduction = np.ones_like(envelope)
            mask = envelope > threshold_linear
            gain_reduction[mask] = threshold_linear / (envelope[mask] + 1e-10)
            gain_reduction = np.clip(gain_reduction, 0.5, 1.0)
            gain_reduction = np.convolve(gain_reduction, np.ones(window_size)/window_size, mode='same')
            sos_low = signal.butter(4, de_ess_freq / nyquist, btype='low', output='sos')
            for ch in range(audio.shape[1]):
                low_content = signal.sosfilt(sos_low, audio[:, ch])
                high_content = signal.sosfilt(sos, audio[:, ch])
                audio[:, ch] = low_content + high_content * gain_reduction
    # EQ
    bass_boost = profile.get("bass_boost_db", 1.0) * (0.5 if light_processing else 1)
    presence_boost = profile.get("presence_boost_db", 1.0) * (0.5 if light_processing else 1)
    if era_info["is_harsh"]:
        presence_boost = max(0, presence_boost - 0.5)
    air_boost = (1.5 if era_info["era"] == "70s_analog" else 0.5) * (0.5 if light_processing else 1)
    if not (light_processing and mastered_check["freq_balance_ok"]):
        eq_chain = [LowShelfFilter(cutoff_frequency_hz=100, gain_db=bass_boost, q=0.7),
                    PeakFilter(cutoff_frequency_hz=300, gain_db=-1.0 if not light_processing else -0.5, q=1.0),
                    PeakFilter(cutoff_frequency_hz=3000, gain_db=presence_boost, q=0.8),
                    HighShelfFilter(cutoff_frequency_hz=12000, gain_db=air_boost, q=0.7)]
        audio = Pedalboard(eq_chain)(audio, sample_rate)
    # Compression
    if not (light_processing and mastered_check["dynamics_ok"]):
        comp_ratio = profile.get("compression_ratio", 2.5) * (0.7 if light_processing else 1)
        comp_threshold = profile.get("compression_threshold", -20) - (3 if light_processing else 0)
        audio = Pedalboard([Compressor(threshold_db=comp_threshold, ratio=max(1.5, comp_ratio),
                                       attack_ms=profile.get("attack_ms", 15), release_ms=profile.get("release_ms", 150))])(audio, sample_rate)
    # Stereo
    width = profile.get("stereo_width", 1.0)
    if era_info["is_narrow_stereo"] and width > 1.1:
        width = 1.1
    if width != 1.0 and audio.shape[1] == 2:
        mid = (audio[:, 0] + audio[:, 1]) / 2
        side = (audio[:, 0] - audio[:, 1]) / 2 * width
        audio[:, 0], audio[:, 1] = mid + side, mid - side
    # Warmth
    if profile.get("add_warmth", False):
        audio = np.tanh(audio * 1.05) / 1.05
    # Loudness
    current_lufs = measure_lufs(audio, sample_rate)
    gain_db = np.clip(profile.get("target_lufs", -14.0) - current_lufs, -6, 9)
    audio = audio * (10 ** (gain_db / 20))
    # Limiting
    ceiling_db = profile.get("ceiling_db", -0.5)
    audio = Pedalboard([Limiter(threshold_db=ceiling_db, release_ms=100)])(audio, sample_rate)
    audio = np.clip(audio, -(10 ** (ceiling_db / 20)), 10 ** (ceiling_db / 20))
    # Export
    stem = file_path.stem
    output_path = file_path.parent / f"{stem}_mastered.mp3"
    temp_wav = file_path.parent / f"{stem}_temp.wav"
    sf.write(str(temp_wav), audio, sample_rate)
    subprocess.run(["ffmpeg", "-y", "-i", str(temp_wav), "-codec:a", "libmp3lame", "-b:a", "320k", str(output_path)], capture_output=True)
    temp_wav.unlink()
    return output_path, "light_touch" if light_processing else "mastered"


def master_files(output_dir: Path, profile: str, verbose: bool = False, remaster: bool = False, force: bool = False):
    """Master all MP3 files in directory."""
    if remaster:
        existing = list(output_dir.glob("*_mastered.mp3"))
        if existing:
            console.print(f"[yellow]Removing {len(existing)} existing mastered files...[/yellow]")
            for f in existing:
                f.unlink()
    mp3_files = [f for f in output_dir.glob("*.mp3") if "_mastered" not in f.stem]
    if not mp3_files:
        console.print("[yellow]No files to master.[/yellow]")
        return
    profile_info = MASTER_PROFILES[profile]
    console.print(f"\n[bold blue]Mastering {len(mp3_files)} files[/bold blue]")
    console.print(f"[dim]Profile: {profile} - {profile_info['description']}[/dim]\n")
    stats = {"mastered": 0, "light_touch": 0, "skipped_already_good": 0, "error": 0}
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=30), TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Mastering", total=len(mp3_files))
        for file_path in mp3_files:
            display = file_path.stem[:45] + "..." if len(file_path.stem) > 45 else file_path.stem
            progress.update(task, description=f"[cyan]{display}")
            try:
                _, status = master_audio(file_path, profile, verbose, force=force or remaster)
                stats[status] = stats.get(status, 0) + 1
                symbol = {"skipped_already_good": "⊘", "light_touch": "~", "mastered": "✓"}.get(status, "✓")
                color = {"skipped_already_good": "dim", "light_touch": "blue", "mastered": "green"}.get(status, "green")
                progress.console.print(f"  [{color}]{symbol}[/{color}] {display}")
            except Exception as e:
                progress.console.print(f"  [red]✗[/red] {display}: {e}")
                stats["error"] += 1
            progress.advance(task)
    console.print(f"\n[bold]Mastering Summary:[/bold]")
    console.print(f"  [green]✓ Full mastering:[/green] {stats['mastered']}")
    console.print(f"  [blue]~ Light touch:[/blue] {stats['light_touch']}")
    console.print(f"  [dim]⊘ Already good:[/dim] {stats['skipped_already_good']}")
    if stats['error']:
        console.print(f"  [red]✗ Errors:[/red] {stats['error']}")


def parse_timestamp(ts) -> float:
    """Convert timestamp string like '1:30' to seconds."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = str(ts).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def analyze_song_audio(file_path: Path, yaml_data: dict = None) -> SongAnalysis:
    """Perform deep audio analysis on a song file."""
    import librosa
    y, sr = librosa.load(str(file_path), sr=22050, mono=True)
    duration_sec = len(y) / sr
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    downbeats = beat_times[::4].tolist() if len(beat_times) >= 4 else beat_times.tolist()
    rms = librosa.feature.rms(y=y)[0]
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, k=8)
        bound_times = librosa.frames_to_time(bounds, sr=sr)
        segments = []
        for i in range(len(bound_times) - 1):
            start, end = bound_times[i], bound_times[i + 1]
            start_frame, end_frame = int(start * sr / 512), min(int(end * sr / 512), len(rms))
            seg_energy = np.mean(rms[start_frame:end_frame]) if end_frame > start_frame else 0
            avg_energy = np.mean(rms)
            label = "chorus" if seg_energy > avg_energy * 1.3 else ("verse_quiet" if seg_energy < avg_energy * 0.7 else "verse")
            segments.append((start, end, label, float(seg_energy)))
    except:
        seg_len = duration_sec / 6
        segments = [(i * seg_len, (i + 1) * seg_len, "unknown", 0.5) for i in range(6)]
    # Intensity on 1-10 scale
    avg_rms = float(np.mean(rms))
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_brightness = float(np.mean(spectral_cent))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = float(np.mean(zcr))
    energy_score = min(avg_rms / 0.15, 1.0)
    brightness_score = min(avg_brightness / 4000, 1.0)
    percussion_score = min(avg_zcr / 0.15, 1.0)
    intensity_raw = (energy_score * 0.5) + (brightness_score * 0.25) + (percussion_score * 0.25)
    intensity = max(1, min(10, int(intensity_raw * 10) + 1))
    good_in, good_out = [], []
    for seg_start, seg_end, label, energy in segments:
        if downbeats:
            nearest = min(downbeats, key=lambda x: abs(x - seg_start))
            if abs(nearest - seg_start) < 1.0:
                good_in.append(nearest)
    for i, (seg_start, seg_end, label, energy) in enumerate(segments[:-1]):
        if i + 1 < len(segments) and energy > segments[i + 1][3]:
            if downbeats:
                good_out.append(min(downbeats, key=lambda x: abs(x - seg_end)))
    best_start, best_end = 0.0, min(90.0, duration_sec)
    if segments:
        best_seg = max(segments, key=lambda x: x[3])
        best_start, best_end = best_seg[0], min(best_seg[1] + 30, duration_sec)
    analysis = SongAnalysis(
        file_path=file_path, name=yaml_data.get("name_en", yaml_data.get("name", file_path.stem)) if yaml_data else file_path.stem,
        artist=yaml_data.get("user_en", yaml_data.get("user", "Unknown")) if yaml_data else "Unknown",
        duration_sec=duration_sec, bpm=bpm, beat_frames=beat_frames, beat_times=beat_times,
        downbeats=downbeats, segments=segments, energy_curve=rms, intensity=intensity,
        best_start_sec=best_start, best_end_sec=best_end, good_transition_in=good_in[:5], good_transition_out=good_out[:5])
    if yaml_data:
        yaml_bpm = yaml_data.get("bpm")
        if yaml_bpm:
            try:
                analysis.yaml_bpm = float(str(yaml_bpm).replace("~", ""))
            except:
                pass
        analysis.yaml_timestamps = yaml_data.get("timestamps", {})
        analysis.yaml_best_sections = yaml_data.get("best_sections", [])
        if analysis.yaml_best_sections:
            first = analysis.yaml_best_sections[0]
            analysis.best_start_sec = parse_timestamp(first.get("start", 0))
            analysis.best_end_sec = parse_timestamp(first.get("end", 90))
    return analysis


def order_songs_by_bpm_curve(analyses: list[SongAnalysis], curve: str, anchor_bpm: float = None) -> list[SongAnalysis]:
    """Order songs according to BPM curve strategy, optionally starting from anchor BPM."""
    if curve == "none" or not analyses:
        return analyses
    sorted_by_bpm = sorted(analyses, key=lambda a: a.bpm)
    n = len(sorted_by_bpm)
    if curve == "up":
        # If anchor has high BPM, we need songs that go UP to it, so return ascending
        return sorted_by_bpm
    elif curve == "down":
        # Descending BPM
        return sorted_by_bpm[::-1]
    elif curve == "peak":
        # Build up to peak in middle, then wind down
        mid = n // 2
        first_half = sorted_by_bpm[:mid]  # ascending
        second_half = sorted_by_bpm[mid:][::-1]  # descending
        return first_half + second_half
    elif curve == "valley":
        # Start high, dip in middle, end high
        mid = n // 2
        first_half = sorted_by_bpm[mid:][::-1]  # descending from fast
        second_half = sorted_by_bpm[:mid][::-1]  # ascending to fast
        return first_half + second_half
    elif curve == "smooth":
        # Greedy nearest-neighbor to minimize BPM jumps
        # Start from the song closest to anchor_bpm, or middle if no anchor
        if not analyses:
            return analyses
        remaining = sorted_by_bpm.copy()
        if anchor_bpm:
            # Find song closest to anchor's BPM to start the chain
            start_idx = min(range(len(remaining)), key=lambda i: abs(remaining[i].bpm - anchor_bpm))
        else:
            start_idx = len(remaining) // 2
        result = [remaining.pop(start_idx)]
        while remaining:
            last_bpm = result[-1].bpm
            nearest_idx = min(range(len(remaining)), key=lambda i: abs(remaining[i].bpm - last_bpm))
            result.append(remaining.pop(nearest_idx))
        return result
    return analyses


def time_stretch_audio(audio: AudioSegment, original_bpm: float, target_bpm: float) -> AudioSegment:
    """Time-stretch audio to match target BPM."""
    if abs(original_bpm - target_bpm) < 1:
        return audio
    try:
        import pyrubberband as pyrb
    except ImportError:
        return audio
    ratio = target_bpm / original_bpm
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    samples = samples / 32768.0
    stretched = pyrb.time_stretch(samples, audio.frame_rate, ratio)
    stretched = (stretched * 32768).astype(np.int16)
    if audio.channels == 2:
        stretched = stretched.flatten()
    return AudioSegment(stretched.tobytes(), frame_rate=audio.frame_rate, sample_width=2, channels=audio.channels)


def extract_section(audio: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    """Extract a section from audio."""
    return audio[int(start_sec * 1000):int(end_sec * 1000)]


def apply_transition(current_audio: AudioSegment, next_audio: AudioSegment, profile: dict,
                     current_analysis: SongAnalysis, next_analysis: SongAnalysis) -> AudioSegment:
    """Apply transition between two audio segments."""
    crossfade_ms = profile["crossfade_ms"]
    if profile["fade_type"] == "hard_cut":
        return current_audio + next_audio
    crossfade_ms = min(len(current_audio), len(next_audio), crossfade_ms)
    if crossfade_ms < 100:
        return current_audio + next_audio
    return current_audio.append(next_audio, crossfade=crossfade_ms)


def find_song_by_name(analyses: list[SongAnalysis], search: str) -> SongAnalysis | None:
    """Find a song by partial name match."""
    search_lower = search.lower()
    for a in analyses:
        full_name = f"{a.artist} - {a.name}".lower()
        if search_lower in full_name or search_lower in a.name.lower() or search_lower in a.artist.lower():
            return a
    return None


def create_mix(output_dir: Path, yaml_data: dict, mix_length: float = 60, avg_song_length: float = 1.5,
               beat_match: bool = True, transition_style: str = "dj", sort_keys: list = None,
               verbose: bool = False, mix_name: str = None, bpm_curve: str = "smooth",
               anchor_song: str = None, anchor_length: float = None) -> Path:
    """Create a DJ mix from downloaded songs."""
    profile = TRANSITION_PROFILES.get(transition_style, TRANSITION_PROFILES["dj"])
    console.print(f"\n[bold blue]Creating Mix[/bold blue]")
    console.print(f"[dim]Length: {mix_length} min | Style: {transition_style} | BPM curve: {bpm_curve}[/dim]")
    if anchor_song:
        console.print(f"[dim]Anchor song: '{anchor_song}' for {anchor_length or avg_song_length} min[/dim]")
    console.print(f"[dim]Beat matching: {'Yes' if beat_match else 'No'}[/dim]\n")
    mp3_files = [f for f in output_dir.glob("*.mp3") if "_mastered" not in f.stem and not is_mix_file(f.stem)]
    if not mp3_files:
        console.print("[red]No MP3 files found to mix.[/red]")
        return None
    yaml_lookup = {}
    for source in yaml_data.get("sources", []):
        name_en = source.get("name_en", source.get("name", ""))
        artist_en = source.get("user_en", source.get("user", ""))
        for mp3 in mp3_files:
            if (name_en and name_en.lower() in mp3.stem.lower()) or (artist_en and artist_en.lower() in mp3.stem.lower()):
                yaml_lookup[mp3] = source
                break
    console.print(f"[cyan]Analyzing {len(mp3_files)} songs...[/cyan]")
    analyses = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=30), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Analyzing", total=len(mp3_files))
        for mp3 in mp3_files:
            display = mp3.stem[:40] + "..." if len(mp3.stem) > 40 else mp3.stem
            progress.update(task, description=f"[cyan]{display}")
            try:
                analysis = analyze_song_audio(mp3, yaml_lookup.get(mp3, {}))
                analyses.append(analysis)
                if verbose:
                    progress.console.print(f"  [green]✓[/green] {display} - {analysis.bpm:.0f} BPM, intensity {analysis.intensity}/10")
            except Exception as e:
                progress.console.print(f"  [red]✗[/red] {display}: {e}")
            progress.advance(task)
    if not analyses:
        console.print("[red]No songs could be analyzed.[/red]")
        return None
    # Handle anchor song
    anchor_analysis = None
    anchor_duration = (anchor_length or avg_song_length) * 60
    if anchor_song:
        anchor_analysis = find_song_by_name(analyses, anchor_song)
        if anchor_analysis:
            analyses.remove(anchor_analysis)
            console.print(f"[green]✓[/green] Anchor song found: {anchor_analysis.artist} - {anchor_analysis.name}")
        else:
            console.print(f"[yellow]⚠[/yellow] Anchor song '{anchor_song}' not found, will skip anchor")
    # Apply BPM curve ordering (use anchor BPM as reference point if present)
    anchor_bpm_ref = anchor_analysis.bpm if anchor_analysis else None
    if bpm_curve != "none":
        analyses = order_songs_by_bpm_curve(analyses, bpm_curve, anchor_bpm=anchor_bpm_ref)
        console.print(f"[dim]Ordered {len(analyses)} songs by BPM curve '{bpm_curve}'[/dim]")
    # Also apply sort_keys if specified (secondary sort)
    if sort_keys:
        def sort_key(a):
            return tuple(a.bpm if k == "bpm" else a.intensity for k in sort_keys)
        analyses.sort(key=sort_key)
    # Calculate how many songs we need (accounting for anchor if present)
    remaining_length = mix_length - (anchor_length or avg_song_length if anchor_analysis else 0)
    target_songs = max(1, int(remaining_length / avg_song_length))
    target_songs = min(target_songs, len(analyses))
    analyses = analyses[:target_songs]
    console.print(f"\n[cyan]Mixing {len(analyses) + (1 if anchor_analysis else 0)} songs...[/cyan]")
    
    # Build the full song list
    all_songs = ([anchor_analysis] if anchor_analysis else []) + analyses
    song_lengths = [anchor_duration if i == 0 and anchor_analysis else avg_song_length * 60 for i in range(len(all_songs))]
    
    # For beat matching: use gradual transitions, not a single target BPM
    # Each song matches to its predecessor for smooth flow
    if beat_match:
        console.print(f"[dim]Beat matching: gradual transitions (neighbor matching)[/dim]")
    
    mix = None
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=30), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Building mix", total=len(all_songs))
        for i, analysis in enumerate(all_songs):
            display = f"{analysis.artist} - {analysis.name}"[:45]
            progress.update(task, description=f"[cyan]{display}")
            try:
                audio = AudioSegment.from_mp3(str(analysis.file_path))
                target_duration = song_lengths[i]
                is_anchor = (i == 0 and anchor_analysis is not None)
                
                # Anchor song always starts from beginning
                if is_anchor:
                    section_start = 0
                else:
                    section_start = analysis.best_start_sec if analysis.best_start_sec > 0 else 0
                section_end = section_start + target_duration
                if section_end > analysis.duration_sec:
                    section_end = analysis.duration_sec
                    if analysis.duration_sec < target_duration:
                        section_start = 0
                    else:
                        section_start = max(0, analysis.duration_sec - target_duration)
                if section_end - section_start < 30:
                    section_start = 0
                    section_end = min(target_duration, analysis.duration_sec)
                segment = extract_section(audio, section_start, section_end)
                
                # Beat matching: anchor plays at natural BPM, others match predecessor
                is_anchor = (i == 0 and anchor_analysis is not None)
                if beat_match and not is_anchor and i > 0:
                    prev_bpm = all_songs[i-1].bpm
                    # Only stretch if BPM difference is significant but not too extreme
                    bpm_diff = abs(analysis.bpm - prev_bpm)
                    if 3 < bpm_diff < 30:  # Don't stretch if too similar or too different
                        # Stretch toward predecessor, but limit to ±15%
                        stretch_ratio = prev_bpm / analysis.bpm
                        stretch_ratio = max(0.85, min(1.15, stretch_ratio))
                        target_bpm_for_song = analysis.bpm * stretch_ratio
                        segment = time_stretch_audio(segment, analysis.bpm, target_bpm_for_song)
                        analysis.bpm = target_bpm_for_song  # Update for next song's reference
                
                segment = segment.normalize()
                actual_duration = len(segment) / 1000
                if mix is None:
                    mix = segment.fade_in(2000)
                else:
                    prev = all_songs[i-1] if i > 0 else None
                    mix = apply_transition(mix, segment, profile, prev, analysis)
                anchor_label = " [ANCHOR - from start]" if is_anchor else ""
                prev_bpm_str = f"{all_songs[i-1].bpm:.0f}→" if i > 0 and beat_match else ""
                progress.console.print(f"  [green]✓[/green] {display} ({actual_duration:.0f}s @ {prev_bpm_str}{analysis.bpm:.0f}bpm){anchor_label}")
            except Exception as e:
                progress.console.print(f"  [red]✗[/red] {display}: {e}")
            progress.advance(task)
    if mix is None:
        console.print("[red]Mix creation failed.[/red]")
        return None
    mix = mix.fade_out(3000)
    output_filename = mix_name or f"mix_{transition_style}_{int(mix_length)}min"
    mix_path = output_dir / f"{output_filename}.mp3"
    console.print(f"\n[cyan]Exporting mix...[/cyan]")
    mix.export(str(mix_path), format="mp3", bitrate="320k")
    actual_length = len(mix) / 1000 / 60
    console.print(f"\n[green]✓ Mix created![/green]")
    console.print(f"  File: [cyan]{mix_path.name}[/cyan]")
    console.print(f"  Length: [cyan]{actual_length:.1f} minutes[/cyan]")
    console.print(f"  Songs: [cyan]{len(all_songs)}[/cyan]")
    # Show BPM progression
    console.print(f"\n[dim]BPM progression:[/dim]")
    bpm_str = " → ".join([f"{a.bpm:.0f}" for a in all_songs[:10]])
    if len(all_songs) > 10:
        bpm_str += f" → ... ({len(all_songs) - 10} more)"
    console.print(f"  [dim]{bpm_str}[/dim]")
    return mix_path


def analyze_audio(file_path: Path) -> dict:
    """Analyze audio file for BPM, intensity (1-10), and other metrics."""
    import librosa
    try:
        y, sr = librosa.load(str(file_path), sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo)))
        rms = librosa.feature.rms(y=y)[0]
        avg_rms, max_rms = float(np.mean(rms)), float(np.max(rms))
        spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_brightness = float(np.mean(spectral_cent))
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = float(np.mean(zcr))
        energy_score = min(avg_rms / 0.15, 1.0)
        brightness_score = min(avg_brightness / 4000, 1.0)
        percussion_score = min(avg_zcr / 0.15, 1.0)
        intensity_raw = (energy_score * 0.5) + (brightness_score * 0.25) + (percussion_score * 0.25)
        intensity = max(1, min(10, int(intensity_raw * 10) + 1))
        return {"bpm": bpm, "intensity": intensity, "intensity_label": INTENSITY_LABELS[intensity],
                "rms": avg_rms, "brightness": avg_brightness, "zcr": avg_zcr, "intensity_raw": intensity_raw}
    except Exception as e:
        return {"error": str(e)}


def rename_with_sort_prefix(file_path: Path, analysis: dict, sort_keys: list[str]) -> Path:
    """Rename file with sort prefix based on analysis."""
    if "error" in analysis:
        return file_path
    prefix_parts = []
    for key in sort_keys:
        if key == "bpm":
            prefix_parts.append(f"{analysis['bpm']:03d}")
        elif key == "intensity":
            prefix_parts.append(analysis["intensity_label"])
        elif key == "brightness":
            prefix_parts.append(f"{int(analysis['brightness']):05d}")
    if not prefix_parts:
        return file_path
    new_name = f"{'-'.join(prefix_parts)}-{file_path.name}"
    new_path = file_path.parent / new_name
    file_path.rename(new_path)
    return new_path


def is_mix_file(filename: str) -> bool:
    """Check if file is a generated mix file."""
    name_lower = filename.lower()
    if "_mix" in name_lower or "mix_" in name_lower:
        return True
    if re.match(r'^\d+-mix', name_lower):
        return True
    return name_lower.endswith('min') and any(c.isdigit() for c in name_lower)


def is_already_prefixed(filename: str) -> bool:
    """Check if file already has our sort prefix."""
    return bool(re.match(r'^(\d{3}-|\d{2}-|\d{3}-\d{2}-)', filename))


def analyze_and_sort_files(output_dir: Path, sort_keys: list[str], verbose: bool = False):
    """Analyze all MP3 files and rename with sort prefixes."""
    mp3_files = [f for f in output_dir.glob("*.mp3") if not is_already_prefixed(f.name)]
    if not mp3_files:
        console.print("[yellow]No new files to analyze.[/yellow]")
        return
    console.print(f"\n[bold blue]Analyzing {len(mp3_files)} files...[/bold blue]\n")
    results = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=30), TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Analyzing audio", total=len(mp3_files))
        for file_path in mp3_files:
            display = file_path.stem[:50] + "..." if len(file_path.stem) > 50 else file_path.stem
            progress.update(task, description=f"[cyan]{display}")
            analysis = analyze_audio(file_path)
            if "error" in analysis:
                progress.console.print(f"  [red]✗[/red] {display}: {analysis['error']}")
            else:
                new_path = rename_with_sort_prefix(file_path, analysis, sort_keys)
                results.append({"name": file_path.stem, "new_name": new_path.name, **analysis})
                if verbose:
                    progress.console.print(f"  [green]✓[/green] {display} → BPM:{analysis['bpm']} Intensity:{analysis['intensity']}/10")
            progress.advance(task)
    if results:
        console.print()
        table = Table(title="Analysis Results", border_style="blue")
        table.add_column("Song", style="cyan", max_width=40)
        table.add_column("BPM", justify="right")
        table.add_column("Intensity", justify="center")
        table.add_column("New Filename", style="dim", max_width=50)
        results.sort(key=lambda r: tuple(r["bpm"] if k == "bpm" else r["intensity"] for k in sort_keys))
        for r in results:
            table.add_row(r["name"][:40], str(r["bpm"]), f"{r['intensity']}/10", r["new_name"][:50])
        console.print(table)


def check_dependency(name: str) -> bool:
    return shutil.which(name) is not None


def get_ytdlp_version() -> str | None:
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None


def sanitize_filename(name: str) -> str:
    for char in '<>:"/\\|?*':
        name = name.replace(char, "")
    return name.strip()


def diagnose_error(stderr: str, stdout: str) -> str:
    combined = f"{stderr} {stdout}".lower()
    patterns = {"video unavailable": "Video deleted/private", "private video": "Video is private",
                "sign in to confirm": "Bot detection (use cookies)", "403": "Forbidden (try cookies)",
                "no js runtime": "Deno not found", "deno": "Deno runtime issue"}
    for pattern, diagnosis in patterns.items():
        if pattern in combined:
            return diagnosis
    for line in stderr.split("\n"):
        if "error" in line.lower():
            return line.strip()[:100]
    return f"Unknown: {stderr[:200]}" if stderr else "Unknown error"


def download_audio(url: str, output_name: str, output_dir: Path, archive_file: Path,
                   cookies_from: str | None = None, verbose: bool = False) -> tuple[bool, str, bool]:
    safe_name = sanitize_filename(output_name)
    output_dir = output_dir.resolve()
    output_template = str(output_dir / f"{safe_name}.%(ext)s")
    final_path = output_dir / f"{safe_name}.mp3"
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "192K",
           "--postprocessor-args", "ffmpeg:-ab 192k", "-o", output_template, "--no-playlist",
           "--embed-thumbnail", "--add-metadata", "--progress", "--newline",
           "--no-check-certificates", "--download-archive", str(archive_file)]
    if cookies_from:
        cmd.extend(["--cookies-from-browser", cookies_from])
    if verbose:
        cmd.append("--verbose")
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if "has already been recorded in the archive" in result.stdout + result.stderr:
            return True, "", True
        if result.returncode == 0:
            if final_path.exists():
                return True, "", False
            for ext in [".m4a", ".webm", ".opus", ".wav"]:
                if (output_dir / f"{safe_name}{ext}").exists():
                    return False, f"Not converted to MP3 (got {ext})", False
            return True, "", False
        return False, diagnose_error(result.stderr, result.stdout), False
    except subprocess.TimeoutExpired:
        return False, "Timeout (>5min)", False
    except FileNotFoundError:
        return False, "yt-dlp not found", False
    except Exception as e:
        return False, f"Exception: {e}", False


def find_yaml_file(directory: Path) -> Path | None:
    yaml_files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
    if not yaml_files:
        return None
    if len(yaml_files) == 1:
        return yaml_files[0]
    console.print("\n[yellow]Multiple YAML files found:[/yellow]")
    for i, f in enumerate(yaml_files):
        console.print(f"  {i + 1}. {f.name}")
    choice = console.input("Select number (Enter for first): ").strip()
    return yaml_files[int(choice) - 1] if choice else yaml_files[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download audio from YouTube URLs in YAML",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--file", type=str, help="YAML file path")
    parser.add_argument("-c", "--cookies", type=str, metavar="BROWSER")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sort_by", type=str, nargs="+", metavar="KEY")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--master", type=str, nargs="?", const="commercial", metavar="PROFILE")
    parser.add_argument("--master_only", action="store_true")
    parser.add_argument("--remaster", action="store_true")
    parser.add_argument("--force_master", action="store_true")
    parser.add_argument("--create_mix", action="store_true")
    parser.add_argument("--mix_length", type=float, default=60)
    parser.add_argument("--avg_song_length", type=float, default=1.5)
    parser.add_argument("--beat_match", action="store_true", default=True)
    parser.add_argument("--no_beat_match", action="store_true")
    parser.add_argument("--transition_style", type=str, default="dj", choices=["dj", "smooth", "radio", "drop"])
    parser.add_argument("--mix_only", action="store_true")
    parser.add_argument("--mix_name", type=str, default=None)
    parser.add_argument("--bpm_curve", type=str, default="smooth", choices=list(BPM_CURVES.keys()),
                        help="BPM progression: up, down, peak, valley, smooth, none")
    parser.add_argument("--anchor_song", type=str, default=None, help="First song (partial name match)")
    parser.add_argument("--anchor_length", type=float, default=None, help="Anchor song duration in minutes")
    args = parser.parse_args()
    if args.no_beat_match:
        args.beat_match = False
    console.print(Panel.fit("[bold blue]YouTube Audio Downloader[/bold blue]", border_style="blue"))
    errors = []
    if not check_dependency("ffmpeg"):
        errors.append(("ffmpeg", "Required for MP3", ["brew install ffmpeg"]))
    else:
        console.print("[green]✓[/green] ffmpeg found")
    if not check_dependency("deno"):
        errors.append(("deno", "Required for YouTube", ["brew install deno"]))
    else:
        console.print("[green]✓[/green] deno found")
    if (args.create_mix or args.mix_only) and args.beat_match:
        if not check_dependency("rubberband"):
            console.print("[yellow]⚠[/yellow] rubberband not found - beat matching disabled")
            args.beat_match = False
        else:
            try:
                import pyrubberband
                console.print("[green]✓[/green] rubberband found")
            except ImportError:
                console.print("[yellow]⚠[/yellow] pyrubberband not found - beat matching disabled")
                args.beat_match = False
    ytdlp_version = get_ytdlp_version()
    if ytdlp_version:
        console.print(f"[green]✓[/green] yt-dlp: [cyan]{ytdlp_version}[/cyan]")
    else:
        errors.append(("yt-dlp", "YouTube downloader", ["pip install -U 'yt-dlp[default]'"]))
    if errors:
        console.print(Panel("\n".join([f"[red]✗ {n}[/red] - {d}\n  [cyan]{c[0]}[/cyan]" for n, d, c in errors]),
                            title="Missing Dependencies", border_style="red"))
        sys.exit(1)
    if args.cookies:
        console.print(f"[green]✓[/green] Cookies: [cyan]{args.cookies}[/cyan]")
    if args.sort_by:
        for key in args.sort_by:
            if key not in {"bpm", "intensity", "brightness"}:
                console.print(f"[red]Invalid sort key '{key}'[/red]")
                sys.exit(1)
    if args.create_mix or args.mix_only:
        console.print(f"[green]✓[/green] Mix: {args.mix_length}min, {args.transition_style}, BPM curve: {args.bpm_curve}")
        if args.anchor_song:
            console.print(f"[green]✓[/green] Anchor: '{args.anchor_song}' for {args.anchor_length or args.avg_song_length}min")
    current_dir = Path.cwd()
    yaml_file = Path(args.file) if args.file else find_yaml_file(current_dir)
    if not yaml_file or (args.file and not yaml_file.exists()):
        console.print(f"[red]Error: YAML file not found[/red]")
        sys.exit(1)
    console.print(f"[green]✓[/green] Using: [cyan]{yaml_file.name}[/cyan]")
    output_dir = current_dir / yaml_file.stem
    output_dir.mkdir(exist_ok=True)
    console.print(f"[green]✓[/green] Output: [cyan]{output_dir.name}/[/cyan]")
    archive_file = output_dir / ".downloaded_archive.txt"
    if args.force and archive_file.exists():
        archive_file.unlink()
        console.print("[yellow]![/yellow] Force: cleared archive")
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sources = data.get("sources", [])
    if not sources:
        console.print("[red]No sources in YAML[/red]")
        sys.exit(1)
    console.print(f"[green]✓[/green] Found [bold]{len(sources)}[/bold] sources\n")
    if args.analyze_only:
        if not args.sort_by:
            console.print("[red]--analyze_only requires --sort_by[/red]")
            sys.exit(1)
        analyze_and_sort_files(output_dir, args.sort_by, args.verbose)
        return
    if args.master_only:
        master_files(output_dir, args.master or "commercial", args.verbose, args.remaster, args.force_master)
        return
    if args.mix_only:
        create_mix(output_dir, data, args.mix_length, args.avg_song_length, args.beat_match,
                   args.transition_style, args.sort_by, args.verbose, args.mix_name,
                   args.bpm_curve, args.anchor_song, args.anchor_length)
        return
    successful, skipped, failed_items = 0, 0, []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=30), TaskProgressColumn(), TimeElapsedColumn(),
                  console=console, transient=False) as progress:
        overall = progress.add_task("[cyan]Overall", total=len(sources))
        for source in sources:
            name = source.get("name_en") or source.get("name", "Unknown")
            artist = source.get("user_en") or source.get("user", "Unknown")
            url = source.get("url")
            output_name = f"{artist} - {name}"
            display = output_name[:50] + "..." if len(output_name) > 50 else output_name
            progress.update(overall, description=f"[cyan]{display}")
            if not url:
                progress.console.print(f"  [yellow]⊘[/yellow] {display} (no URL)")
                failed_items.append({"name": name, "artist": artist, "url": "N/A", "reason": "No URL"})
                progress.advance(overall)
                continue
            success, error_msg, was_skipped = download_audio(url, output_name, output_dir, archive_file, args.cookies, args.verbose)
            if was_skipped:
                progress.console.print(f"  [dim]⊘[/dim] {display}")
                skipped += 1
            elif success:
                progress.console.print(f"  [green]✓[/green] {display}")
                successful += 1
            else:
                progress.console.print(f"  [red]✗[/red] {display}: {error_msg.split(chr(10))[0][:80]}")
                failed_items.append({"name": name, "artist": artist, "url": url, "reason": error_msg.split('\n')[0][:80]})
            progress.advance(overall)
    console.print()
    summary = Table(title="Summary", show_header=False, border_style="blue")
    summary.add_column("Label", style="bold")
    summary.add_column("Value")
    summary.add_row("✓ Successful", f"[green]{successful}[/green]")
    summary.add_row("⊘ Skipped", f"[dim]{skipped}[/dim]")
    summary.add_row("✗ Failed", f"[red]{len(failed_items)}[/red]")
    console.print(summary)
    if args.sort_by:
        analyze_and_sort_files(output_dir, args.sort_by, args.verbose)
    if args.master:
        master_files(output_dir, args.master, args.verbose, args.remaster, args.force_master)
    if args.create_mix:
        create_mix(output_dir, data, args.mix_length, args.avg_song_length, args.beat_match,
                   args.transition_style, args.sort_by, args.verbose, args.mix_name,
                   args.bpm_curve, args.anchor_song, args.anchor_length)


if __name__ == "__main__":
    main()