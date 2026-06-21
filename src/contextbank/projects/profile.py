from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROFILE_RELATIVE_PATH = ".contextbank/project.yml"
DEFAULT_AUTOLOAD_TOKEN_BUDGET = 1500
DEFAULT_AUTOLOAD_OUTPUT_TYPE = "implementation"


@dataclass(frozen=True)
class ProjectAutoloadSettings:
    enabled: bool = True
    token_budget: int = DEFAULT_AUTOLOAD_TOKEN_BUDGET
    output_type: str = DEFAULT_AUTOLOAD_OUTPUT_TYPE

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "token_budget": self.token_budget,
            "output_type": self.output_type,
        }


@dataclass(frozen=True)
class ProjectProfile:
    name: str
    goals: list[str] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    exclude_topics: list[str] = field(default_factory=list)
    autoload: ProjectAutoloadSettings = field(default_factory=ProjectAutoloadSettings)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "goals": self.goals,
            "stack": self.stack,
            "topics": self.topics,
            "constraints": self.constraints,
            "exclude_topics": self.exclude_topics,
            "autoload": self.autoload.to_dict(),
        }


def profile_path(project_path: str | Path = ".") -> Path:
    return Path(project_path).expanduser().resolve() / PROFILE_RELATIVE_PATH


def init_project_profile(
    project_path: str | Path = ".",
    *,
    name: str | None = None,
    goals: list[str] | None = None,
    stack: list[str] | None = None,
    topics: list[str] | None = None,
    constraints: list[str] | None = None,
    exclude_topics: list[str] | None = None,
    autoload_enabled: bool = True,
    autoload_token_budget: int = DEFAULT_AUTOLOAD_TOKEN_BUDGET,
    autoload_output_type: str = DEFAULT_AUTOLOAD_OUTPUT_TYPE,
    overwrite: bool = False,
) -> Path:
    root = Path(project_path).expanduser().resolve()
    path = profile_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    profile = ProjectProfile(
        name=name or root.name,
        goals=goals or [],
        stack=stack or [],
        topics=topics or [],
        constraints=constraints or [],
        exclude_topics=exclude_topics or [],
        autoload=ProjectAutoloadSettings(
            enabled=autoload_enabled,
            token_budget=_valid_token_budget(
                autoload_token_budget,
                default=DEFAULT_AUTOLOAD_TOKEN_BUDGET,
            ),
            output_type=autoload_output_type or DEFAULT_AUTOLOAD_OUTPUT_TYPE,
        ),
    )
    path.write_text(render_project_profile(profile), encoding="utf-8")
    return path


def save_project_profile(project_path: str | Path, profile: ProjectProfile) -> Path:
    path = profile_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_project_profile(profile), encoding="utf-8")
    return path


def configure_project_autoload(
    project_path: str | Path = ".",
    *,
    enabled: bool,
    token_budget: int | None = None,
    output_type: str | None = None,
) -> Path:
    root = Path(project_path).expanduser().resolve()
    profile = load_project_profile(root) or ProjectProfile(name=root.name)
    current = profile.autoload
    updated = ProjectProfile(
        name=profile.name,
        goals=profile.goals,
        stack=profile.stack,
        topics=profile.topics,
        constraints=profile.constraints,
        exclude_topics=profile.exclude_topics,
        autoload=ProjectAutoloadSettings(
            enabled=enabled,
            token_budget=_valid_token_budget(
                token_budget if token_budget is not None else current.token_budget,
                default=current.token_budget,
            ),
            output_type=(output_type or current.output_type or DEFAULT_AUTOLOAD_OUTPUT_TYPE),
        ),
    )
    return save_project_profile(root, updated)


def load_project_profile(project_path: str | Path = ".") -> ProjectProfile | None:
    path = profile_path(project_path)
    if not path.exists():
        return None
    return parse_project_profile(path.read_text(encoding="utf-8"))


def inspect_project(project_path: str | Path = ".") -> dict[str, object]:
    root = Path(project_path).expanduser().resolve()
    profile = load_project_profile(root)
    autoload = profile.autoload if profile else ProjectAutoloadSettings()
    return {
        "root_path": str(root),
        "profile_path": str(profile_path(root)),
        "exists": profile is not None,
        "profile": profile.to_dict() if profile else None,
        "autoload": {
            **autoload.to_dict(),
            "source": "profile" if profile else "default",
        },
    }


def render_project_profile(profile: ProjectProfile) -> str:
    lines = [f"name: {profile.name}"]
    for key in ("goals", "stack", "topics", "constraints", "exclude_topics"):
        values = getattr(profile, key)
        lines.append(f"{key}:")
        if values:
            lines.extend(f"  - {value}" for value in values)
        else:
            lines.append("  []")
    lines.append("autoload:")
    lines.append(f"  enabled: {_format_bool(profile.autoload.enabled)}")
    lines.append(f"  token_budget: {profile.autoload.token_budget}")
    lines.append(f"  output_type: {profile.autoload.output_type}")
    return "\n".join(lines) + "\n"


def parse_project_profile(text: str) -> ProjectProfile:
    name = "project"
    lists: dict[str, list[str]] = {
        "goals": [],
        "stack": [],
        "topics": [],
        "constraints": [],
        "exclude_topics": [],
    }
    autoload = ProjectAutoloadSettings()
    current_list: str | None = None
    current_map: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = _strip_scalar(value)
            current_list = key if key in lists else None
            current_map = key if key == "autoload" else None
            if key == "name" and value:
                name = value
            elif key == "autoload" and value:
                autoload = ProjectAutoloadSettings(
                    enabled=_parse_bool(value, default=autoload.enabled),
                    token_budget=autoload.token_budget,
                    output_type=autoload.output_type,
                )
            continue
        stripped = line.strip()
        if current_list and stripped.startswith("- "):
            lists[current_list].append(_strip_scalar(stripped[2:]))
            continue
        if current_map == "autoload" and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = _strip_scalar(value)
            if key == "enabled":
                autoload = ProjectAutoloadSettings(
                    enabled=_parse_bool(value, default=autoload.enabled),
                    token_budget=autoload.token_budget,
                    output_type=autoload.output_type,
                )
            elif key == "token_budget":
                autoload = ProjectAutoloadSettings(
                    enabled=autoload.enabled,
                    token_budget=_valid_token_budget(
                        _parse_int(value, default=autoload.token_budget),
                        default=autoload.token_budget,
                    ),
                    output_type=autoload.output_type,
                )
            elif key == "output_type" and value:
                autoload = ProjectAutoloadSettings(
                    enabled=autoload.enabled,
                    token_budget=autoload.token_budget,
                    output_type=value,
                )
    return ProjectProfile(name=name, autoload=autoload, **lists)


def _strip_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _parse_bool(value: str, *, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "on", "1", "enabled"}:
        return True
    if normalized in {"false", "no", "off", "0", "disabled"}:
        return False
    return default


def _parse_int(value: str, *, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _valid_token_budget(value: int, *, default: int) -> int:
    return value if value >= 250 else default


def _format_bool(value: bool) -> str:
    return "true" if value else "false"
