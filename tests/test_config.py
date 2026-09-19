"""
Tests for archive-root resolution.

The archive deliberately lives outside the repository, so where it lives is
environment configuration. STATION_ARCHIVE_ROOT is the name to use;
DELOS_STATION_ARCHIVE is honoured for continuity with the project this
package was extracted from.
"""
from __future__ import annotations

from pathlib import Path

from observations import config


def test_preferred_env_var_is_used(monkeypatch, tmp_path):
    monkeypatch.delenv("DELOS_STATION_ARCHIVE", raising=False)
    monkeypatch.setenv("STATION_ARCHIVE_ROOT", str(tmp_path / "chosen"))
    assert config.archive_root() == tmp_path / "chosen"


def test_legacy_env_var_still_works(monkeypatch, tmp_path):
    monkeypatch.delenv("STATION_ARCHIVE_ROOT", raising=False)
    monkeypatch.setenv("DELOS_STATION_ARCHIVE", str(tmp_path / "legacy"))
    assert config.archive_root() == tmp_path / "legacy"


def test_preferred_env_var_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("STATION_ARCHIVE_ROOT", str(tmp_path / "chosen"))
    monkeypatch.setenv("DELOS_STATION_ARCHIVE", str(tmp_path / "legacy"))
    assert config.archive_root() == tmp_path / "chosen"


def test_falls_back_to_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("STATION_ARCHIVE_ROOT", raising=False)
    monkeypatch.delenv("DELOS_STATION_ARCHIVE", raising=False)
    assert config.archive_root() == Path(config.DEFAULT_ARCHIVE_ROOT)


def test_empty_env_var_is_ignored(monkeypatch, tmp_path):
    """An exported-but-empty variable must not resolve the archive to ''."""
    monkeypatch.setenv("STATION_ARCHIVE_ROOT", "")
    monkeypatch.setenv("DELOS_STATION_ARCHIVE", str(tmp_path / "legacy"))
    assert config.archive_root() == tmp_path / "legacy"
