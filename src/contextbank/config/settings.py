from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contextbank.paths import ContextBankPaths, secure_write_text

DEFAULT_CONFIG_TEXT = """# ContextBank local-first configuration.

[storage]
database = "contextbank.db"
markdown_mirror = true

[agents]
default_mode = "read-only"

[retrieval]
default_limit = 10

[ai]
mode = "none"
generation_provider = "none"
generation_model = ""
generation_base_url = ""
generation_credential_env = ""
generation_timeout = 60.0
embedding_provider = "none"
embedding_model = ""
embedding_base_url = ""
embedding_credential_env = ""
allow_cloud = false

[x]
client_id = ""
user_id = ""
"""


class StorageSettings(BaseModel):
    database: str = "contextbank.db"
    markdown_mirror: bool = True


class AgentSettings(BaseModel):
    default_mode: str = "read-only"


class RetrievalSettings(BaseModel):
    default_limit: int = Field(default=10, ge=1)


class AISettings(BaseModel):
    mode: str = "none"
    generation_provider: str = "none"
    generation_model: str = ""
    generation_base_url: str = ""
    generation_credential_env: str = ""
    generation_timeout: float = Field(default=60.0, gt=0)
    embedding_provider: str = "none"
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_credential_env: str = ""
    allow_cloud: bool = False


class XSettings(BaseModel):
    client_id: str = ""
    user_id: str = ""


class ContextBankSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    home: Path
    storage: StorageSettings = Field(default_factory=StorageSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ai: AISettings = Field(default_factory=AISettings)
    x: XSettings = Field(default_factory=XSettings)

    @property
    def database_path(self) -> Path:
        configured = Path(self.storage.database).expanduser()
        if configured.is_absolute():
            return configured
        return self.home / configured

    @property
    def paths(self) -> ContextBankPaths:
        base_paths = ContextBankPaths.from_home(self.home)
        if self.database_path == base_paths.database:
            return base_paths
        return ContextBankPaths(
            home=base_paths.home,
            config_file=base_paths.config_file,
            database=self.database_path,
            raw=base_paths.raw,
            documents=base_paths.documents,
            cards=base_paths.cards,
            projects=base_paths.projects,
            exports=base_paths.exports,
            logs=base_paths.logs,
        )


def initialize_settings(
    home: Path | str | None = None,
    *,
    overwrite_config: bool = False,
) -> ContextBankSettings:
    paths = ContextBankPaths.from_home(Path(home).expanduser() if home is not None else None)
    paths.ensure()
    if overwrite_config or not paths.config_file.exists():
        secure_write_text(paths.config_file, DEFAULT_CONFIG_TEXT)
    return load_settings(paths.home)


def load_settings(home: Path | str | None = None) -> ContextBankSettings:
    paths = ContextBankPaths.from_home(Path(home).expanduser() if home is not None else None)
    config_data: dict[str, Any] = {}
    if paths.config_file.exists():
        with paths.config_file.open("rb") as config_file:
            config_data = tomllib.load(config_file)
    return ContextBankSettings(home=paths.home, **config_data)


def ensure_contextbank_home(home: Path | str | None = None) -> ContextBankPaths:
    paths = ContextBankPaths.from_home(Path(home).expanduser() if home is not None else None)
    paths.ensure()
    if not paths.config_file.exists():
        secure_write_text(paths.config_file, DEFAULT_CONFIG_TEXT)
    return paths
