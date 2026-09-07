"""Voice I/O: speech in, speech out, all local.

Nothing here touches a cloud API. Speech-to-text is faster-whisper (a
CPU-friendly CTranslate2 build of Whisper); text-to-speech is Piper (a small
neural voice) with espeak as an always-there fallback. Capture and playback go
through the tools already on the machine - `arecord`/`aplay` over PipeWire -
rather than a native audio binding, so there is no PortAudio to compile and the
same code works over SSH-forwarded audio or a plain ALSA box.

The one non-obvious piece is *when to stop recording*. Fixed-length clips feel
awful, so `listen()` streams raw PCM from arecord and watches the amplitude:
it waits for you to start talking, then stops once you have been quiet for a
short beat. That gives natural "press-to-talk, speak, pause" turns with only
arecord in the pipeline.
"""

from __future__ import annotations

import array
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import deps, paths
from .errors import PAError

SAMPLE_RATE = 16_000  # what Whisper expects
CHANNELS = 1
SAMPLE_WIDTH = 2  # S16_LE


class VoiceError(PAError):
    """Voice capture, transcription, or synthesis failed."""


@dataclass
class ListenResult:
    text: str
    seconds: float
    aborted: bool = False


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _recorder_cmd() -> list[str]:
    """The command that streams raw S16_LE mono PCM to stdout."""
    if shutil.which("parecord"):  # PipeWire/Pulse native, respects default source
        return ["parecord", "--rate", str(SAMPLE_RATE), "--channels", str(CHANNELS),
                "--format", "s16le", "--raw"]
    if shutil.which("arecord"):
        return ["arecord", "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                "-c", str(CHANNELS), "-t", "raw"]
    if shutil.which("ffmpeg"):
        return ["ffmpeg", "-loglevel", "quiet", "-f", "alsa", "-i", "default",
                "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-f", "s16le", "-"]
    raise VoiceError(
        "no audio recorder found. Install one:\n"
        "  sudo apt install alsa-utils   (arecord)  or  pipewire-pulse (parecord)"
    )


def _rms(frame: bytes) -> float:
    """Root-mean-square amplitude of a PCM frame, 0..~32768."""
    if not frame:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class Microphone:
    """Streams PCM from a recorder subprocess with silence-based turn ending."""

    def __init__(
        self,
        *,
        silence_rms: float = 500.0,
        silence_end: float = 1.2,
        max_seconds: float = 30.0,
        start_timeout: float = 10.0,
        min_speech: float = 0.3,
    ) -> None:
        self.silence_rms = silence_rms
        self.silence_end = silence_end
        self.max_seconds = max_seconds
        self.start_timeout = start_timeout
        self.min_speech = min_speech

    def calibrate(self, seconds: float = 0.6) -> float:
        """Sample ambient noise and set the silence threshold above it, so a
        loud room does not read as constant speech."""
        proc = subprocess.Popen(_recorder_cmd(), stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        chunk = SAMPLE_RATE * SAMPLE_WIDTH // 10  # 100ms
        levels = []
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                data = proc.stdout.read(chunk)
                if not data:
                    break
                levels.append(_rms(data))
        finally:
            proc.terminate()
            proc.wait(timeout=2)
        if levels:
            ambient = sorted(levels)[len(levels) // 2]
            self.silence_rms = max(400.0, ambient * 2.5)
        return self.silence_rms

    def record(self, on_state=None) -> bytes:
        """Capture one utterance. Returns raw PCM bytes (may be empty if the
        speaker never started). `on_state` is called with 'waiting'/'listening'
        so the UI can show a prompt."""
        proc = subprocess.Popen(_recorder_cmd(), stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        chunk = SAMPLE_RATE * SAMPLE_WIDTH // 20  # 50ms frames
        frames: list[bytes] = []
        started = False
        speech_frames = 0
        silent_run = 0.0
        frame_seconds = chunk / (SAMPLE_RATE * SAMPLE_WIDTH)
        start_deadline = time.monotonic() + self.start_timeout

        try:
            if on_state:
                on_state("waiting")
            while True:
                data = proc.stdout.read(chunk)
                if not data:
                    break
                level = _rms(data)
                loud = level >= self.silence_rms

                if not started:
                    if loud:
                        started = True
                        speech_frames = 1
                        frames.append(data)
                        if on_state:
                            on_state("listening")
                    elif time.monotonic() > start_deadline:
                        break  # nobody spoke
                    continue

                frames.append(data)
                if loud:
                    speech_frames += 1
                    silent_run = 0.0
                else:
                    silent_run += frame_seconds
                    if silent_run >= self.silence_end:
                        break

                if len(frames) * frame_seconds >= self.max_seconds:
                    break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        if not started or speech_frames * frame_seconds < self.min_speech:
            return b""
        return b"".join(frames)


def pcm_to_wav(pcm: bytes, path: Path) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return path


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------


class Transcriber:
    """faster-whisper, loaded once and reused."""

    def __init__(self, model: str = "base", *, auto_install: bool = True) -> None:
        self.model_name = model
        self._auto = auto_install
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            fw = deps.require(
                "faster_whisper", auto=self._auto, purpose="speech recognition"
            )
            # int8 keeps the base model small and fast on a CPU with no GPU.
            self._model = fw.WhisperModel(
                self.model_name, device="cpu", compute_type="int8"
            )
        return self._model

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        tmp = paths.cache_dir() / f"listen-{int(time.time() * 1000)}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pcm_to_wav(pcm, tmp)
        try:
            model = self._load()
            segments, _ = model.transcribe(str(tmp), language=None, vad_filter=True)
            return " ".join(seg.text for seg in segments).strip()
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------


# Where a Piper voice model might already live, so we can reuse one rather than
# insist on a download. Checked in order.
_VOICE_SEARCH = [
    paths.data_dir() / "voices",
    Path.home() / ".local/share/piper-voices",
    Path.home() / ".local/share/piper",
]


def _find_piper_voice(configured: str | None) -> Path | None:
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None
    for directory in _VOICE_SEARCH:
        if directory.is_dir():
            for onnx in sorted(directory.glob("*.onnx")):
                if (onnx.with_suffix(".onnx.json")).exists() or onnx.with_suffix(".json").exists():
                    return onnx
    return None


class Speaker:
    """Text to speech. Prefers Piper (neural), falls back to espeak (robotic
    but instant and always present)."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        voice_model: str | None = None,
        auto_install: bool = True,
        rate: int = 175,
    ) -> None:
        self.backend = backend
        self.voice_model = voice_model
        self._auto = auto_install
        self.rate = rate
        self._piper: Any = None
        self._resolved: str | None = None

    def resolve(self) -> str:
        """Decide which backend to actually use, once."""
        if self._resolved:
            return self._resolved
        wanted = self.backend
        if wanted in ("auto", "piper"):
            voice = _find_piper_voice(self.voice_model)
            if voice is not None and (deps.available("piper") or self._auto):
                try:
                    piper = deps.require("piper", auto=self._auto, purpose="neural speech")
                    self._piper = piper.PiperVoice.load(str(voice))
                    self._resolved = "piper"
                    return self._resolved
                except Exception:  # noqa: BLE001 - fall through to espeak
                    if wanted == "piper":
                        raise VoiceError(
                            f"could not load Piper voice {voice}; check the model file"
                        )
            elif wanted == "piper":
                raise VoiceError(
                    "backend is 'piper' but no voice model was found. Put a "
                    f"<name>.onnx (+ .onnx.json) in {_VOICE_SEARCH[0]}, or set "
                    "voice.voice_model in the config. Download voices from "
                    "https://huggingface.co/rhasspy/piper-voices"
                )
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            self._resolved = "espeak"
            return self._resolved
        if shutil.which("spd-say"):
            self._resolved = "spd-say"
            return self._resolved
        raise VoiceError(
            "no text-to-speech available. Install one:\n"
            "  sudo apt install espeak-ng   (simple)\n"
            "  or pip install piper-tts and add a voice model (natural)"
        )

    def to_wav(self, text: str, path: Path) -> Path:
        backend = self.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if backend == "piper":
            with wave.open(str(path), "wb") as handle:
                self._piper.synthesize_wav(text, handle)
        else:
            binary = shutil.which("espeak-ng") or shutil.which("espeak")
            subprocess.run(
                [binary, "-s", str(self.rate), "-w", str(path), text],
                check=True, capture_output=True,
            )
        return path

    def say(self, text: str) -> None:
        """Synthesize and play, blocking until playback finishes."""
        text = (text or "").strip()
        if not text:
            return
        backend = self.resolve()

        if backend == "spd-say":
            subprocess.run(["spd-say", "-w", text], check=False)
            return

        wav = paths.cache_dir() / f"say-{int(time.time() * 1000)}.wav"
        try:
            self.to_wav(text, wav)
            self._play(wav)
        finally:
            wav.unlink(missing_ok=True)

    @staticmethod
    def _play(wav: Path) -> None:
        player = (
            shutil.which("paplay") or shutil.which("aplay") or shutil.which("ffplay")
        )
        if not player:
            raise VoiceError("no audio player found (install alsa-utils for aplay)")
        cmd = [player, str(wav)]
        if player.endswith("ffplay"):
            cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav)]
        subprocess.run(cmd, check=False, capture_output=True)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class Voice:
    """Bundles capture, transcription, and speech for the CLI and the speak tool."""

    def __init__(self, config: Any) -> None:
        settings = getattr(config, "voice", None) or {}
        self.enabled_config = settings
        auto = getattr(config, "auto_install_deps", True)
        self.mic = Microphone(
            silence_end=float(settings.get("silence_end", 1.2)),
            max_seconds=float(settings.get("max_seconds", 30.0)),
        )
        self.transcriber = Transcriber(
            settings.get("stt_model", "base"), auto_install=auto
        )
        self.speaker = Speaker(
            backend=settings.get("tts_backend", "auto"),
            voice_model=settings.get("voice_model"),
            auto_install=auto,
            rate=int(settings.get("rate", 175)),
        )

    def listen(self, on_state=None) -> ListenResult:
        start = time.monotonic()
        pcm = self.mic.record(on_state=on_state)
        if not pcm:
            return ListenResult("", time.monotonic() - start, aborted=True)
        text = self.transcriber.transcribe(pcm)
        return ListenResult(text, time.monotonic() - start)

    def speak(self, text: str) -> None:
        self.speaker.say(text)

    def check(self) -> str:
        lines = []
        try:
            _recorder_cmd()
            lines.append("input:  ok")
        except VoiceError as exc:
            lines.append(f"input:  {exc}")
        try:
            backend = self.speaker.resolve()
            lines.append(f"output: {backend}")
        except VoiceError as exc:
            lines.append(f"output: {exc}")
        lines.append(f"stt:    faster-whisper '{self.transcriber.model_name}'")
        return "\n".join(lines)
