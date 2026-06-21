from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOME = Path.home() / ".contextbank"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True)
class ContextBankPaths:
    home: Path
    config_file: Path
    database: Path
    raw: Path
    documents: Path
    cards: Path
    projects: Path
    exports: Path
    logs: Path

    @classmethod
    def from_home(cls, home: Path | None = None) -> ContextBankPaths:
        root = home or Path(os.environ.get("CONTEXTBANK_HOME", DEFAULT_HOME)).expanduser()
        return cls(
            home=root,
            config_file=root / "config.toml",
            database=root / "contextbank.db",
            raw=root / "raw",
            documents=root / "documents",
            cards=root / "cards",
            projects=root / "projects",
            exports=root / "exports",
            logs=root / "logs",
        )

    def ensure(self) -> None:
        for path in (
            self.home,
            self.raw,
            self.raw / "x",
            self.raw / "web",
            self.raw / "files",
            self.documents,
            self.cards,
            self.projects,
            self.exports,
            self.logs,
        ):
            secure_mkdir(path)


def secure_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _chmod_best_effort(path, PRIVATE_DIR_MODE)


def secure_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    secure_mkdir(path.parent)
    path.write_text(text, encoding=encoding)
    _chmod_best_effort(path, PRIVATE_FILE_MODE)


def secure_write_bytes(path: Path, data: bytes) -> None:
    secure_mkdir(path.parent)
    path.write_bytes(data)
    _chmod_best_effort(path, PRIVATE_FILE_MODE)


def secure_file(path: Path) -> None:
    if path.exists():
        _chmod_best_effort(path, PRIVATE_FILE_MODE)


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass
