import pytest

from app.settings import Settings


def test_default_transcription_limits_support_six_hour_recordings(monkeypatch):
    monkeypatch.delenv("MAX_AUDIO_SECONDS", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    for settings in (Settings(), Settings.from_env()):
        assert settings.max_audio_seconds == 21600
        assert settings.max_upload_mb * 1024 * 1024 >= 21600 * 16000 * 2 + 44
        assert settings.max_task_audio_mb >= settings.max_upload_mb


def test_deployment_can_lower_the_duration_limit(monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_SECONDS", "10800")
    assert Settings.from_env().max_audio_seconds == 10800


def test_duration_limit_cannot_exceed_six_hours(monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_SECONDS", "21601")
    with pytest.raises(ValueError, match="MAX_AUDIO_SECONDS.*21600"):
        Settings.from_env()
