from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from shared.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def load_adjacency(config_path: Optional[str | Path]) -> dict[str, list[str]]:
    if not config_path:
        logger.warning(
            "No service-graph config provided — topology features "
            "(Group G) will be neutral (0 upstream/downstream, health=1.0). "
            "Set the service-graph YAML path to enable them."
        )
        return {}

    path = Path(config_path)
    if not path.exists():
        logger.warning("Service-graph config not found at %s — topology features will be neutral.", path)
        return {}

    raw = load_config(str(path))
    services = raw.get("services") if raw else None
    if not services:
        logger.warning("Service-graph config at %s has no 'services' list.", path)
        return {}

    adjacency: dict[str, list[str]] = {}
    for svc in services:
        name = svc.get("name")
        if not name:
            continue
        adjacency[name] = list(svc.get("dependencies", []) or [])

    logger.info("Loaded topology adjacency for %d services from %s", len(adjacency), path)
    return adjacency
