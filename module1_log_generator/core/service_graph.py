from __future__ import annotations

from typing import Iterator

import networkx as nx

from module1_log_generator.models.schema import ServiceGraphConfig


class ServiceGraph:


    def __init__(self, config: ServiceGraphConfig) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._config = config
        self._build(config)

    def _build(self, config: ServiceGraphConfig) -> None:
        for svc in config.services:
            self._graph.add_node(svc.name, config=svc)
            for dep in svc.dependencies:
                # Edge: svc.name → dep  (svc calls dep)
                self._graph.add_edge(svc.name, dep)


    def topological_order(self) -> list[str]:
        
        try:
            return list(reversed(list(nx.topological_sort(self._graph))))
        except nx.NetworkXUnfeasible:
            raise ValueError(
                "Cannot compute topological order on a cyclic graph. "
                "Call detect_cycles() first."
            )

    def get_dependencies(self, service_name: str) -> list[str]:

        return list(self._graph.successors(service_name))

    def get_dependents(self, service_name: str) -> list[str]:

        return list(self._graph.predecessors(service_name))

    def detect_cycles(self) -> list[list[str]]:

        return list(nx.simple_cycles(self._graph))

    def all_services(self) -> list[str]:
        return list(self._graph.nodes)

    def service_config(self, name: str):
        return self._graph.nodes[name]["config"]

    def edge_call_volume(self, source: str, target: str) -> int:
        cfg = self._graph.nodes[source].get("config")
        return cfg.log_volume_per_sec if cfg else 100

    def to_adjacency_dict(self) -> dict[str, list[str]]:
        return {node: list(self._graph.successors(node))
                for node in self._graph.nodes}

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def __contains__(self, service_name: str) -> bool:
        return service_name in self._graph
