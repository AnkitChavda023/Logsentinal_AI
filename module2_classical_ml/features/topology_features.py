from __future__ import annotations


class TopologyFeatureExtractor:

    def __init__(self, adjacency: dict[str, list[str]]) -> None:
        self._adj = adjacency
        # Reverse adjacency: which services call this one
        self._rev: dict[str, list[str]] = {}
        for svc, deps in adjacency.items():
            for dep in deps:
                self._rev.setdefault(dep, []).append(svc)

        # anomaly_scores: service → latest anomaly score (0–1)
        self._health: dict[str, float] = {}

    def update_health(self, service: str, anomaly_score: float) -> None:
        self._health[service] = anomaly_score

    def extract(self, service: str) -> dict[str, float]:
        downstream = self._adj.get(service, [])
        upstream = self._rev.get(service, [])

        upstream_scores = [self._health.get(u, 0.0) for u in upstream]
        # Health = 1 − anomaly_score (higher = healthier)
        upstream_health = (
            1.0 - (sum(upstream_scores) / len(upstream_scores))
            if upstream_scores
            else 1.0
        )

        return {
            "upstream_service_count": float(len(upstream)),
            "downstream_service_count": float(len(downstream)),
            "upstream_health_score": upstream_health,
        }
