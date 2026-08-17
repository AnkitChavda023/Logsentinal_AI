
from __future__ import annotations

from pathlib import Path

import yaml

from module1_log_generator.models.schema import ServiceGraphConfig


def load_and_validate(path: str | Path) -> ServiceGraphConfig:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Service graph config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Empty YAML file: {path}")

    config: ServiceGraphConfig = ServiceGraphConfig.model_validate(raw)

    _validate_dependencies(config)
    _validate_no_cycles(config)

    return config


def _validate_dependencies(config: ServiceGraphConfig) -> None:

    defined = {svc.name for svc in config.services}

    for svc in config.services:
        for dep in svc.dependencies:
            if dep not in defined:
                raise ValueError(
                    f"Service '{svc.name}' lists '{dep}' as a dependency, "
                    f"but '{dep}' is not defined in this service graph. "
                    f"Defined services: {sorted(defined)}"
                )


def _validate_no_cycles(config: ServiceGraphConfig) -> None:

    if config.allow_cycles:
        return

    import networkx as nx
    graph = nx.DiGraph()
    for svc in config.services:
        graph.add_node(svc.name)
        for dep in svc.dependencies:
            graph.add_edge(svc.name, dep)

    try:
        list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        # Find the actual cycle to report
        try:
            cycle = nx.find_cycle(graph)
            cycle_str = " → ".join([u for u, v in cycle] + [cycle[0][0]])
        except nx.NetworkXNoCycle:
            cycle_str = "Unknown"

        raise ValueError(
            f"Cycle detected in service graph: {cycle_str}. "
            f"Set 'allow_cycles: true' in your YAML to bypass this check."
        )
