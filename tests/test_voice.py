"""Voice tests that need no live microphone.

The recorder subprocess is replaced with one that emits synthetic PCM, so the
silence-detection turn logic is exercised against real amplitude data. The
transcription and playback backends are only checked for correct *selection*
and error messages - actually running Whisper or a speaker belongs in the
manual/live check, not the unit suite.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pa import config as config_mod, voice as voice_mod
from pa.voice import Microphone, Speaker, VoiceError, _rms, pcm_to_wav


def test_rms_distinguishes_silence_from_signal():
    silence = b"\x00\x00" * 1000
    import struct

    loud = b"".join(struct.pack("<h", 8000 if i % 2 else -8000) for i in range(1000))
    assert _rms(silence) == 0.0
    assert _rms(loud) > 5000


def test_pcm_to_wav_roundtrips(tmp_path):
    pcm = b"\x01\x02" * 8000  # 1s at 16k mono
    path = pcm_to_wav(pcm, tmp_path / "x.wav")
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.readframes(handle.getnframes()) == pcm


def _fake_recorder(pcm: bytes):
    """A recorder command that streams the given PCM bytes then stops."""
    import base64

    encoded = base64.b64encode(pcm).decode()
    return [
        sys.executable, "-c",
        f"import sys,base64,time; "
        f"sys.stdout.buffer.write(base64.b64decode('{encoded}')); "
        f"sys.stdout.buffer.flush(); time.sleep(0.2)",
    ]


def test_record_returns_nothing_on_pure_silence(monkeypatch):
    silence = b"\x00\x00" * 16000  # 1s of dead quiet
    monkeypatch.setattr(voice_mod, "_recorder_cmd", lambda: _fake_recorder(silence))
    mic = Microphone(start_timeout=0.5, silence_rms=500)
    assert mic.record() == b""


def test_record_captures_speech_then_stops_on_silence(monkeypatch):
    import struct

    def tone(samples, amp):
        return b"".join(struct.pack("<h", amp if i % 2 else -amp) for i in range(samples))

    # 0.5s loud, then 1.5s quiet - should capture and stop after the pause.
    pcm = tone(8000, 6000) + tone(24000, 0)
    monkeypatch.setattr(voice_mod, "_recorder_cmd", lambda: _fake_recorder(pcm))
    mic = Microphone(silence_rms=500, silence_end=1.0, min_speech=0.2)
    captured = mic.record()
    assert len(captured) > 0
    # It stopped near the end of speech, not after consuming all 2s of quiet.
    assert len(captured) < len(pcm)


def test_speaker_falls_back_to_espeak_when_no_voice_model(monkeypatch):
    # No piper model anywhere, but espeak present.
    monkeypatch.setattr(voice_mod, "_find_piper_voice", lambda cfg: None)
    monkeypatch.setattr(
        voice_mod.shutil, "which",
        lambda name: "/usr/bin/espeak" if name in ("espeak", "espeak-ng") else None,
    )
    speaker = Speaker(backend="auto", auto_install=False)
    assert speaker.resolve() == "espeak"


def test_speaker_piper_backend_without_model_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(voice_mod, "_find_piper_voice", lambda cfg: None)
    speaker = Speaker(backend="piper", auto_install=False)
    with pytest.raises(VoiceError, match="no voice model"):
        speaker.resolve()


def test_speaker_errors_when_nothing_available(monkeypatch):
    monkeypatch.setattr(voice_mod, "_find_piper_voice", lambda cfg: None)
    monkeypatch.setattr(voice_mod.shutil, "which", lambda name: None)
    speaker = Speaker(backend="auto", auto_install=False)
    with pytest.raises(VoiceError, match="no text-to-speech"):
        speaker.resolve()


def test_voice_facade_check_reports_each_stage():
    report = voice_mod.Voice(config_mod.default_config()).check()
    assert "input:" in report and "output:" in report and "stt:" in report
