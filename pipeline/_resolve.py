"""
_resolve.py — layered config resolution shared across tune.py, app.py tabs,
and any future pipeline stage.

Resolution hierarchy for any setting (most specific wins):
    1. video_overrides.{video_name}.{key}
    2. batches.{batch_name}.{key}        (the batch containing this video, if any)
    3. {key} at project level
    4. (not handled here) value in pipeline/default.settings

A video must not appear in more than one batch. resolve_batch_for_video
returns the first match and logs a warning if more than one batch claims
the same video.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_batch_for_video(video_name: str, project_config: dict) -> str | None:
    """
    Return the name of the batch containing video_name, or None if the
    video is not listed in any batch.
    """
    batches = project_config.get("batches") or {}
    matches = [
        batch_name for batch_name, batch_config in batches.items()
        if video_name in (batch_config.get("videos") or [])
    ]
    if len(matches) > 1:
        logger.warning(
            "Video '%s' appears in multiple batches (%s) — using the first one.",
            video_name, matches,
        )
    return matches[0] if matches else None


def resolve_effective_config(video_name: str, project_config: dict) -> dict:
    """
    Build the effective config for a video by layering:
        project-level → batch (if any) → video_overrides (if any)

    Each layer's keys overwrite the previous layer's. Returns a new dict;
    project_config itself is not modified.
    """
    effective_config = dict(project_config)

    batch_name = resolve_batch_for_video(video_name, project_config)
    if batch_name is not None:
        batch_config = dict(project_config["batches"][batch_name])
        batch_config.pop("videos", None)  # not a setting, just membership list
        if batch_config:
            logger.info(
                "Applying batch '%s' settings for video '%s': %s",
                batch_name, video_name, batch_config,
            )
        effective_config.update(batch_config)

    video_overrides = (project_config.get("video_overrides") or {}).get(video_name, {})
    if video_overrides:
        logger.info(
            "Applying per-video overrides for '%s': %s", video_name, video_overrides
        )
        effective_config.update(video_overrides)

    return effective_config
