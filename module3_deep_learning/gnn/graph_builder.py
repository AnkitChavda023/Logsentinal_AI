"""
Builds the PyTorch Geometric graph from the service dependency dict.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def build_service_graph(
    adjacency: dict[str, list[str]],
    service_index: dict[str, int],
    node_features: torch.Tensor | None = None,
    edge_features: dict[tuple[int, int], list[float]] | None = None,
) -> Data:
    """
    Build a PyTorch Geometric Data object from a service adjacency dict.

    Args:
        adjacency:     {service: [deps]} — "service calls dep" = directed edge.
        service_index: {service_name: node_index}
        node_features: Optional (num_services, feature_dim) tensor.
        edge_features: Optional {(src, dst): [features]} dict.

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr.
    """
    n = len(service_index)

    # Build edge_index: edge A → B means A depends on B
    edges_src, edges_dst = [], []
    edge_attrs = []

    for src_name, deps in adjacency.items():
        src_idx = service_index.get(src_name)
        if src_idx is None:
            continue
        for dep_name in deps:
            dst_idx = service_index.get(dep_name)
            if dst_idx is None:
                continue
            edges_src.append(src_idx)
            edges_dst.append(dst_idx)
            ef = edge_features.get((src_idx, dst_idx), [1.0]) if edge_features else [1.0]
            edge_attrs.append(ef)

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    x = (
        node_features
        if node_features is not None
        else torch.zeros(n, 256, dtype=torch.float32)
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
