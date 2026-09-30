"""
Tests for archive-root resolution (settings.py).

The archive deliberately lives outside the repository, so where it lives is
environment configuration, and STATION_ARCHIVE_ROOT is the one variable that
sets it. A second name was honoured here once; these tests no longer have to
guard against it.
"""
from __future__ import annotations

from pathlib import Path

from station_archive import settings


def test_the_env_var_sets_the_archive_root(monkeypatch, tmp_path):
    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path / "chosen"))
    assert settings.archive_root() == tmp_path / "chosen"


def test_falls_back_to_the_default_when_unset(monkeypatch):
    monkeypatch.delenv(settings.ARCHIVE_ROOT_ENV_VAR, raising=False)
    assert settings.archive_root() == Path(settings.default_archive_root())


def test_empty_env_var_is_ignored(monkeypatch):
    """
    An exported-but-empty variable must not resolve the archive to '', which
    would put it in the current working directory -- normally the checkout.
    """
    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, "")
    assert settings.archive_root() == settings.default_archive_root()


def test_default_root_is_absolute_on_this_platform(monkeypatch):
    """
    A bare drive-letter default is absolute on Windows and RELATIVE on POSIX,
    where it would create ./E:/station_archive inside the checkout -- in a
    repository whose ignore rules exist to keep licensed provider data out of
    git. The default must be absolute wherever it resolves.
    """
    monkeypatch.delenv(settings.ARCHIVE_ROOT_ENV_VAR, raising=False)
    assert settings.default_archive_root().is_absolute()
    assert settings.archive_root().is_absolute()


def test_posix_default_is_not_a_drive_letter_path():
    from pathlib import PurePosixPath

    assert not PurePosixPath("E:/station_archive").is_absolute()


def test_the_default_belongs_to_whoever_runs_it(monkeypatch):
    """
    The Windows default used to be a hardcoded drive-letter path -- the drive of
    the machine this package was extracted from. Every Windows checkout then
    resolved the archive to a path that almost certainly did not exist, and
    nothing said so until a command failed.

    The guarantee now is that the default sits under the running user's home
    directory, whoever that is, on every platform.
    """
    monkeypatch.delenv(settings.ARCHIVE_ROOT_ENV_VAR, raising=False)
    default = settings.default_archive_root()
    assert default.is_relative_to(Path.home())
    assert not default.drive or default.drive == Path.home().drive


def test_user_agent_reports_the_running_version():
    """A provider log has to be able to say which build made a request."""
    from station_archive import __version__

    assert __version__ in settings.USER_AGENT
