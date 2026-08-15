"""
_parameters.py — single source of truth for the TRex parameters exposed in
the app's Tune and Parameters tabs.

Each parameter is declared once here and drives everything downstream:

  * the Tune tab's sweep controls
  * the Parameters tab's project-level and per-batch fields
  * the overrides written into each run's .settings file (_sweep.py)
  * reading values back out of a clip's .settings file

Adding a parameter means adding one entry here, not editing five files.

Resolution is unchanged: video_overrides -> batch -> project -> default.settings.
A parameter left unset anywhere is simply not written as an override, so
whatever is in default.settings governs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    """
    name        key used in project.yaml, batches, and video_overrides
                (identical to the TRex setting name, to avoid a mapping layer)
    label       shown in the UI
    kind        "bool" | "int" | "float"
    default     TRex's own default, shown as a hint when nothing is set
    help        one-line explanation shown under the field
    depends_on  name of a bool parameter that must be true for this to apply
    """
    name:       str
    label:      str
    kind:       str
    default:    object
    help:       str = ""
    depends_on: str | None = None

    def parse(self, raw: str):
        """Convert a raw form value to its typed Python value, or None if blank."""
        if raw is None:
            return None
        text = str(raw).strip()
        if text == "":
            return None
        if self.kind == "bool":
            return text.lower() in ("true", "1", "yes", "on")
        if self.kind == "int":
            return int(float(text))   # tolerate "3.0" from number inputs
        return float(text)


# =============================================================================
# Detection parameters — affect how blobs are found (TGrabs/detection stage)
# =============================================================================

DETECTION_PARAMETERS = [
    Parameter(
        name="use_closing", label="Use closing", kind="bool", default=False,
        help="Try to 'close' irregular blobs before detection.",
    ),
    Parameter(
        name="closing_size", label="Closing size", kind="int", default=3,
        help="Size of the dilation/erosion filters used for closing.",
        depends_on="use_closing",
    ),
    Parameter(
        name="dilation_size", label="Dilation size", kind="float", default=0.0,
        help=">0 inflates shapes (may merge them); <0 shrinks them (may split "
             "them). As a fraction of the initial size.",
    ),
]


# =============================================================================
# Tracking parameters — affect how blobs are linked into tracks
# =============================================================================

TRACKING_PARAMETERS = [
    Parameter(
        name="track_max_individuals", label="Max individuals", kind="int", default=8,
        help="Expected number of individuals.",
    ),
    Parameter(
        name="track_max_speed", label="Max speed (px/s)", kind="float", default=4000.0,
        help="A track is split if the animal appears to move faster than this.",
    ),
    Parameter(
        name="track_max_reassign_time", label="Max reassign time (s)",
        kind="float", default=10.0,
        help="After this long, the tracker stops trying to reidentify an "
             "individual by location, and assigns a new ID.",
    ),
    Parameter(
        name="track_history_split_threshold", label="History split threshold (s)",
        kind="float", default=200.0,
        help="A blob detected for at least this long will not be split into "
             "multiple blobs.",
    ),
]


ALL_PARAMETERS = DETECTION_PARAMETERS + TRACKING_PARAMETERS
PARAMETERS_BY_NAME = {parameter.name: parameter for parameter in ALL_PARAMETERS}


# =============================================================================
# Helpers
# =============================================================================

def collect_overrides(effective_config: dict) -> dict:
    """
    Return {trex_setting: value} for every exposed parameter that is actually
    set in the resolved config. Parameters left unset are omitted, so
    default.settings governs them.

    A parameter whose depends_on gate is false is also omitted — writing
    closing_size while use_closing is false would be misleading in the
    exported .settings file.
    """
    overrides = {}
    for parameter in ALL_PARAMETERS:
        value = effective_config.get(parameter.name)
        if value is None:
            continue
        if parameter.depends_on and not effective_config.get(parameter.depends_on):
            continue
        overrides[parameter.name] = value
    return overrides


def read_from_settings(settings: dict) -> dict:
    """
    Extract exposed parameter values from a parsed .settings file
    ({key: raw_string}), typed according to each parameter's kind.
    Missing or unparseable entries are returned as None.
    """
    values = {}
    for parameter in ALL_PARAMETERS:
        raw = settings.get(parameter.name)
        try:
            values[parameter.name] = parameter.parse(raw)
        except (TypeError, ValueError):
            values[parameter.name] = None
    return values
