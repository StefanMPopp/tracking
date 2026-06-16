"""
Utilities for reading and patching TRex .settings files.

Format: one "parameter = value" per line. Comments (#) are dropped on read.
"""

from pathlib import Path


def read_settings(settings_file: Path) -> dict:
    """Parse a .settings file into an ordered dict of {key: raw_value_string}."""
    settings = {}
    for line in settings_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        settings[key.strip()] = value.strip()
    return settings


def write_settings(settings: dict, output_file: Path) -> None:
    """Write an ordered dict of settings to a .settings file."""
    lines = [f"{key} = {value}" for key, value in settings.items()]
    output_file.write_text("\n".join(lines) + "\n")


def patch_settings(
    base_settings_file: Path,
    output_file: Path,
    overrides: dict,
) -> None:
    """
    Read base_settings_file, apply overrides, write result to output_file.

    Values are converted to TRex string format automatically.
    Lists are written as TRex expects: [0,150] or [[90,24000]].
    """
    settings = read_settings(base_settings_file)
    for key, value in overrides.items():
        settings[key] = _to_trex_string(value)
    write_settings(settings, output_file)


def _to_trex_string(value) -> str:
    """Convert a Python value to its TRex settings string representation."""
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            inner = ",".join(
                "[" + ",".join(str(v) for v in inner_list) + "]"
                for inner_list in value
            )
            return f"[{inner}]"
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)
