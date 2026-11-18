"""
Root Cause Tracer - Causal Chain Reconstruction Module

Implements the causal chain reconstruction algorithm from Section 7.8:
1. Start at predicted failing service (the "symptom")
2. Look up its dependencies in the graph
3. For each dependency: is it also anomalous (per Module 2)?
4. If yes: recurse further upstream
5. Stop at a service that is anomalous but whose dependencies are all healthy
6. That service = ROOT CAUSE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

import networkx as nx
import numpy as np

from module3_deep_learning.config import defaults as cfg

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Types of evidence supporting causal chain links."""
    ANOMALY_SCORE = "anomaly_score"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE = "error_rate"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    CONNECTION_ISSUE = "connection_issue"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_PRESSURE = "cpu_pressure"


@dataclass
class CausalEvidence:
    """Evidence supporting a link in the causal chain."""
    service: str
    timestamp_offset_seconds: float
    evidence_type: EvidenceType
    evidence_text: str
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service": self.service,
            "timestamp_offset_s": self.timestamp_offset_seconds,
            "evidence": self.evidence_text,
            "evidence_type": self.evidence_type.value,
            "confidence": self.confidence,
            "metrics": self.metrics
        }


@dataclass
class CausalChainLink:
    """A single link in the causal chain."""
    service: str
    anomaly_score: float
    failure_probability: float
    timestamp_offset_seconds: float
    evidence: List[CausalEvidence]
    is_root_cause: bool = False
    depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service": self.service,
            "anomaly_score": self.anomaly_score,
            "failure_probability": self.failure_probability,
            "timestamp_offset_s": self.timestamp_offset_seconds,
            "evidence": [e.to_dict() for e in self.evidence],
            "is_root_cause": self.is_root_cause,
            "depth": self.depth
        }


@dataclass
class CausalChainResult:
    """Complete causal chain analysis result."""
    symptom_service: str
    root_cause_service: str
    root_cause_confidence: float
    chain: List[CausalChainLink]
    alternative_causes: List[Tuple[str, float]]  # (service, confidence)
    analysis_timestamp: datetime
    total_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symptom_service": self.symptom_service,
            "root_cause_service": self.root_cause_service,
            "root_cause_confidence": self.root_cause_confidence,
            "causal_chain": [link.to_dict() for link in self.chain],
            "alternative_causes": [
                {"service": s, "confidence": c} 
                for s, c in self.alternative_causes
            ],
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_depth": self.total_depth
        }


@dataclass
class ServiceHealthState:
    """Current health state of a service."""
    service_name: str
    anomaly_score: float  # From Module 2 (0-1, higher = more anomalous)
    failure_probability: float  # From Module 3 prediction
    latency_ms: float
    error_rate: float
    cpu_usage: float
    memory_mb: float
    last_error_timestamp: Optional[datetime] = None
    active_connections: Optional[int] = None
    queue_depth: Optional[int] = None
    
    @property
    def is_anomalous(self) -> bool:
        """Check if service is in anomalous state (§6.3 WARNING/ANOMALY threshold)."""
        return self.anomaly_score >= cfg.ANOMALY_THRESHOLD
    
    @property
    def is_critical(self) -> bool:
        """Check if service is in critical anomalous state (§6.3 ANOMALY threshold)."""
        return self.anomaly_score >= cfg.CRITICAL_ANOMALY_THRESHOLD


class RootCauseTracer:
    """
    Traces causal chains through the service dependency graph to identify root causes.
    
    Implements the algorithm from Section 7.8:
    - Starts at the predicted failing service (symptom)
    - Traverses upstream through dependencies
    - Identifies the root cause as the anomalous service with healthy dependencies
    """
    
    def __init__(
        self,
        dependency_graph: nx.DiGraph,
        anomaly_threshold: float = cfg.ANOMALY_THRESHOLD,
        critical_threshold: float = cfg.CRITICAL_ANOMALY_THRESHOLD,
        max_depth: int = cfg.ROOT_CAUSE_MAX_DEPTH,
        min_root_cause_confidence: float = cfg.ROOT_CAUSE_MIN_CONFIDENCE
    ):
        """
        Initialize the root cause tracer.
        
        Args:
            dependency_graph: NetworkX DiGraph where edges represent "A calls B"
            anomaly_threshold: Minimum anomaly score to consider a service anomalous
            critical_threshold: Threshold for critical anomaly classification
            max_depth: Maximum depth to traverse in causal chain
            min_root_cause_confidence: Minimum confidence to report a root cause
        """
        self.graph = dependency_graph
        self.anomaly_threshold = anomaly_threshold
        self.critical_threshold = critical_threshold
        self.max_depth = max_depth
        self.min_root_cause_confidence = min_root_cause_confidence
        
        # Validate graph structure
        if not isinstance(self.graph, nx.DiGraph):
            raise ValueError("dependency_graph must be a NetworkX DiGraph")
        
        logger.info(
            f"RootCauseTracer initialized with {self.graph.number_of_nodes()} services, "
            f"{self.graph.number_of_edges()} dependencies"
        )
    
    def trace_root_cause(
        self,
        symptom_service: str,
        service_health_states: Dict[str, ServiceHealthState],
        prediction_timestamp: Optional[datetime] = None
    ) -> CausalChainResult:
        """
        Trace the causal chain from a failing service to its root cause.
        
        Args:
            symptom_service: The service showing failure symptoms
            service_health_states: Current health state of all services
            prediction_timestamp: Timestamp of the prediction (for evidence timeline)
            
        Returns:
            CausalChainResult containing the full causal chain analysis
        """
        if prediction_timestamp is None:
            prediction_timestamp = datetime.utcnow()
        
        if symptom_service not in self.graph:
            raise ValueError(f"Service '{symptom_service}' not found in dependency graph")
        
        if symptom_service not in service_health_states:
            raise ValueError(f"No health state provided for service '{symptom_service}'")
        
        logger.debug(f"Starting root cause trace from symptom service: {symptom_service}")
        
        # Track visited services to avoid cycles
        visited: Set[str] = set()
        
        # Build the causal chain
        chain: List[CausalChainLink] = []
        alternative_causes: List[Tuple[str, float]] = []
        
        # Start recursive traversal
        root_cause_service, root_cause_confidence = self._trace_upstream(
            current_service=symptom_service,
            service_health_states=service_health_states,
            visited=visited,
            chain=chain,
            alternative_causes=alternative_causes,
            depth=0,
            prediction_timestamp=prediction_timestamp
        )
        
        # If no clear root cause found, the symptom service is the root cause
        if root_cause_service is None:
            root_cause_service = symptom_service
            root_cause_confidence = service_health_states[symptom_service].anomaly_score
        
        # Mark the root cause in the chain
        for link in chain:
            if link.service == root_cause_service:
                link.is_root_cause = True
        
        # Sort chain by depth (root cause first)
        chain.sort(key=lambda x: -x.depth)
        
        # Sort alternative causes by confidence
        alternative_causes.sort(key=lambda x: -x[1])
        
        result = CausalChainResult(
            symptom_service=symptom_service,
            root_cause_service=root_cause_service,
            root_cause_confidence=root_cause_confidence,
            chain=chain,
            alternative_causes=alternative_causes[:5],  # Top 5 alternatives
            analysis_timestamp=prediction_timestamp,
            total_depth=len(chain)
        )
        
        logger.info(
            f"Root cause trace complete: {symptom_service} -> {root_cause_service} "
            f"(confidence: {root_cause_confidence:.2f}, chain depth: {len(chain)})"
        )
        
        return result
    
    def _trace_upstream(
        self,
        current_service: str,
        service_health_states: Dict[str, ServiceHealthState],
        visited: Set[str],
        chain: List[CausalChainLink],
        alternative_causes: List[Tuple[str, float]],
        depth: int,
        prediction_timestamp: datetime
    ) -> Tuple[Optional[str], float]:
        """
        Recursively trace upstream through dependencies to find root cause.
        
        Returns:
            Tuple of (root_cause_service, confidence) or (None, 0.0) if not found
        """
        if depth > self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached at service {current_service}")
            return None, 0.0
        
        if current_service in visited:
            logger.debug(f"Cycle detected at service {current_service}, skipping")
            return None, 0.0
        
        visited.add(current_service)
        
        # Get current service health
        current_health = service_health_states.get(current_service)
        if current_health is None:
            logger.warning(f"No health state for service {current_service}")
            return None, 0.0
        
        # Build evidence for this service
        evidence = self._build_evidence(current_health, depth, prediction_timestamp)
        
        # Add this service to the chain
        chain_link = CausalChainLink(
            service=current_service,
            anomaly_score=current_health.anomaly_score,
            failure_probability=current_health.failure_probability,
            timestamp_offset_seconds=-depth * 60,  # Approximate offset
            evidence=evidence,
            is_root_cause=False,
            depth=depth
        )
        chain.append(chain_link)
        
        # If this service is not anomalous, it's not in the causal chain
        if not current_health.is_anomalous:
            logger.debug(f"Service {current_service} is healthy, not in causal chain")
            chain.pop()  # Remove from chain since it's healthy
            return None, 0.0
        
        # Get dependencies (services this one calls)
        dependencies = self._get_dependencies(current_service)
        
        if not dependencies:
            # No dependencies and anomalous = this is the root cause
            logger.debug(f"Service {current_service} has no dependencies and is anomalous - ROOT CAUSE")
            return current_service, current_health.anomaly_score
        
        # Check if all dependencies are healthy
        anomalous_dependencies: List[Tuple[str, float]] = []
        
        for dep_service in dependencies:
            dep_health = service_health_states.get(dep_service)
            if dep_health is None:
                continue
            
            if dep_health.is_anomalous:
                anomalous_dependencies.append((dep_service, dep_health.anomaly_score))
        
        # If all dependencies are healthy, this service is the root cause
        if not anomalous_dependencies:
            logger.debug(
                f"Service {current_service} is anomalous but all {len(dependencies)} "
                f"dependencies are healthy - ROOT CAUSE"
            )
            return current_service, current_health.anomaly_score
        
        # Sort anomalous dependencies by anomaly score (highest first)
        anomalous_dependencies.sort(key=lambda x: -x[1])
        
        # Add lower-priority dependencies as alternative causes
        for dep_service, score in anomalous_dependencies[1:]:
            alternative_causes.append((dep_service, score))
        
        # Recurse into the most anomalous dependency
        primary_dep = anomalous_dependencies[0][0]
        root_cause, confidence = self._trace_upstream(
            current_service=primary_dep,
            service_health_states=service_health_states,
            visited=visited,
            chain=chain,
            alternative_causes=alternative_causes,
            depth=depth + 1,
            prediction_timestamp=prediction_timestamp
        )
        
        # If no root cause found in dependencies, this service is the root cause
        if root_cause is None:
            return current_service, current_health.anomaly_score
        
        return root_cause, confidence
    
    def _get_dependencies(self, service: str) -> List[str]:
        """Get the services that the given service depends on (calls)."""
        if service not in self.graph:
            return []
        
        # In the graph, edges go from caller to callee
        # So successors are the dependencies
        return list(self.graph.successors(service))
    
    def _get_dependents(self, service: str) -> List[str]:
        """Get the services that depend on (call) the given service."""
        if service not in self.graph:
            return []
        
        # Predecessors are the services that call this one
        return list(self.graph.predecessors(service))
    
    def _build_evidence(
        self,
        health: ServiceHealthState,
        depth: int,
        prediction_timestamp: datetime
    ) -> List[CausalEvidence]:
        """Build evidence list for a service based on its health state."""
        evidence: List[CausalEvidence] = []
        timestamp_offset = -depth * 60  # Approximate, more recent = less negative
        
        # Check for high anomaly score
        if health.anomaly_score >= self.critical_threshold:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.ANOMALY_SCORE,
                evidence_text=f"Critical anomaly score: {health.anomaly_score:.2f}",
                confidence=health.anomaly_score,
                metrics={"anomaly_score": health.anomaly_score}
            ))
        elif health.anomaly_score >= self.anomaly_threshold:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.ANOMALY_SCORE,
                evidence_text=f"Elevated anomaly score: {health.anomaly_score:.2f}",
                confidence=health.anomaly_score,
                metrics={"anomaly_score": health.anomaly_score}
            ))
        
        # Check for latency issues (assuming normal is < 200ms for most services)
        if health.latency_ms > 1000:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.LATENCY_SPIKE,
                evidence_text=f"Severe latency spike: {health.latency_ms:.0f}ms",
                confidence=min(health.latency_ms / 5000, 1.0),
                metrics={"latency_ms": health.latency_ms}
            ))
        elif health.latency_ms > 500:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.LATENCY_SPIKE,
                evidence_text=f"Elevated latency: {health.latency_ms:.0f}ms",
                confidence=min(health.latency_ms / 2000, 1.0),
                metrics={"latency_ms": health.latency_ms}
            ))
        
        # Check for high error rate
        if health.error_rate > 0.10:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.ERROR_RATE,
                evidence_text=f"High error rate: {health.error_rate*100:.1f}%",
                confidence=min(health.error_rate * 5, 1.0),
                metrics={"error_rate": health.error_rate}
            ))
        elif health.error_rate > 0.05:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.ERROR_RATE,
                evidence_text=f"Elevated error rate: {health.error_rate*100:.1f}%",
                confidence=min(health.error_rate * 3, 1.0),
                metrics={"error_rate": health.error_rate}
            ))
        
        # Check for CPU pressure
        if health.cpu_usage > 0.90:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.CPU_PRESSURE,
                evidence_text=f"CPU near saturation: {health.cpu_usage*100:.0f}%",
                confidence=health.cpu_usage,
                metrics={"cpu_usage": health.cpu_usage}
            ))
        elif health.cpu_usage > 0.80:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.CPU_PRESSURE,
                evidence_text=f"High CPU usage: {health.cpu_usage*100:.0f}%",
                confidence=health.cpu_usage * 0.9,
                metrics={"cpu_usage": health.cpu_usage}
            ))
        
        # Check for memory pressure (assuming 4096 MB limit)
        memory_ratio = health.memory_mb / 4096
        if memory_ratio > 0.90:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.MEMORY_PRESSURE,
                evidence_text=f"Memory near limit: {health.memory_mb:.0f}MB",
                confidence=memory_ratio,
                metrics={"memory_mb": health.memory_mb}
            ))
        elif memory_ratio > 0.80:
            evidence.append(CausalEvidence(
                service=health.service_name,
                timestamp_offset_seconds=timestamp_offset,
                evidence_type=EvidenceType.MEMORY_PRESSURE,
                evidence_text=f"High memory usage: {health.memory_mb:.0f}MB",
                confidence=memory_ratio * 0.9,
                metrics={"memory_mb": health.memory_mb}
            ))
        
        # Check for connection pool issues
        if health.active_connections is not None:
            # Assuming max 100 connections
            conn_ratio = health.active_connections / 100
            if conn_ratio > 0.90:
                evidence.append(CausalEvidence(
                    service=health.service_name,
                    timestamp_offset_seconds=timestamp_offset,
                    evidence_type=EvidenceType.RESOURCE_EXHAUSTION,
                    evidence_text=f"Connection pool near exhaustion: {health.active_connections}/100",
                    confidence=conn_ratio,
                    metrics={"active_connections": health.active_connections}
                ))
            elif conn_ratio > 0.80:
                evidence.append(CausalEvidence(
                    service=health.service_name,
                    timestamp_offset_seconds=timestamp_offset,
                    evidence_type=EvidenceType.RESOURCE_EXHAUSTION,
                    evidence_text=f"Connection pool filling: {health.active_connections}/100",
                    confidence=conn_ratio * 0.9,
                    metrics={"active_connections": health.active_connections}
                ))
        
        return evidence
    
    def trace_multiple_symptoms(
        self,
        symptom_services: List[str],
        service_health_states: Dict[str, ServiceHealthState],
        prediction_timestamp: Optional[datetime] = None
    ) -> List[CausalChainResult]:
        """
        Trace root causes for multiple symptom services.
        
        Args:
            symptom_services: List of services showing failure symptoms
            service_health_states: Current health state of all services
            prediction_timestamp: Timestamp of the prediction
            
        Returns:
            List of CausalChainResult for each symptom service
        """
        results: List[CausalChainResult] = []
        
        for service in symptom_services:
            try:
                result = self.trace_root_cause(
                    symptom_service=service,
                    service_health_states=service_health_states,
                    prediction_timestamp=prediction_timestamp
                )
                results.append(result)
            except ValueError as e:
                logger.warning(f"Failed to trace root cause for {service}: {e}")
        
        return results
    
    def identify_common_root_causes(
        self,
        results: List[CausalChainResult]
    ) -> Dict[str, List[str]]:
        """
        Identify common root causes across multiple causal chain results.
        
        Args:
            results: List of causal chain analysis results
            
        Returns:
            Dict mapping root cause service to list of affected symptom services
        """
        root_cause_map: Dict[str, List[str]] = {}
        
        for result in results:
            root_cause = result.root_cause_service
            if root_cause not in root_cause_map:
                root_cause_map[root_cause] = []
            root_cause_map[root_cause].append(result.symptom_service)
        
        return root_cause_map
    
    def compute_blast_radius(
        self,
        root_cause_service: str,
        service_health_states: Dict[str, ServiceHealthState]
    ) -> Dict[str, Any]:
        """
        Compute the potential blast radius of a root cause failure.
        
        Args:
            root_cause_service: The identified root cause service
            service_health_states: Current health state of all services
            
        Returns:
            Dict containing blast radius analysis
        """
        if root_cause_service not in self.graph:
            raise ValueError(f"Service '{root_cause_service}' not found in graph")
        
        # Find all services that transitively depend on the root cause
        affected_services: Set[str] = set()
        self._find_affected_services(root_cause_service, affected_services)
        
        # Calculate impact metrics
        total_services = self.graph.number_of_nodes()
        affected_count = len(affected_services)
        impact_ratio = affected_count / total_services if total_services > 0 else 0
        
        # Get health states of affected services
        affected_health: List[Dict[str, Any]] = []
        for service in affected_services:
            health = service_health_states.get(service)
            if health:
                affected_health.append({
                    "service": service,
                    "anomaly_score": health.anomaly_score,
                    "failure_probability": health.failure_probability,
                    "is_currently_anomalous": health.is_anomalous
                })
        
        # Sort by anomaly score
        affected_health.sort(key=lambda x: -x["anomaly_score"])
        
        return {
            "root_cause_service": root_cause_service,
            "total_affected_services": affected_count,
            "impact_ratio": impact_ratio,
            "affected_services": list(affected_services),
            "affected_health_summary": affected_health,
            "critical_path_services": self._find_critical_path_services(root_cause_service)
        }
    
    def _find_affected_services(
        self,
        service: str,
        affected: Set[str],
        visited: Optional[Set[str]] = None
    ) -> None:
        """Recursively find all services that depend on the given service."""
        if visited is None:
            visited = set()
        
        if service in visited:
            return
        
        visited.add(service)
        
        # Get services that call this service (dependents)
        dependents = self._get_dependents(service)
        
        for dependent in dependents:
            affected.add(dependent)
            self._find_affected_services(dependent, affected, visited)
    
    def _find_critical_path_services(self, service: str) -> List[str]:
        """
        Find services on the critical path (high in-degree or out-degree).
        
        These are services that, if they fail, would affect many others.
        """
        critical_services: List[Tuple[str, int]] = []
        
        for node in self.graph.nodes():
            # Critical score based on total connections
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            critical_score = in_degree + out_degree
            
            if critical_score > 2:  # At least 3 connections
                critical_services.append((node, critical_score))
        
        # Sort by critical score
        critical_services.sort(key=lambda x: -x[1])
        
        return [s[0] for s in critical_services[:5]]  # Top 5


class RootCauseTracerBuilder:
    """Builder for RootCauseTracer with fluent interface."""
    
    def __init__(self):
        self._graph: Optional[nx.DiGraph] = None
        self._anomaly_threshold: float = cfg.ANOMALY_THRESHOLD
        self._critical_threshold: float = cfg.CRITICAL_ANOMALY_THRESHOLD
        self._max_depth: int = cfg.ROOT_CAUSE_MAX_DEPTH
        self._min_confidence: float = 0.5
    
    def with_graph(self, graph: nx.DiGraph) -> "RootCauseTracerBuilder":
        """Set the dependency graph."""
        self._graph = graph
        return self
    
    def with_graph_from_edges(
        self,
        edges: List[Tuple[str, str]]
    ) -> "RootCauseTracerBuilder":
        """Build graph from list of (caller, callee) edges."""
        self._graph = nx.DiGraph()
        self._graph.add_edges_from(edges)
        return self
    
    def with_anomaly_threshold(self, threshold: float) -> "RootCauseTracerBuilder":
        """Set the anomaly detection threshold."""
        self._anomaly_threshold = threshold
        return self
    
    def with_critical_threshold(self, threshold: float) -> "RootCauseTracerBuilder":
        """Set the critical anomaly threshold."""
        self._critical_threshold = threshold
        return self
    
    def with_max_depth(self, depth: int) -> "RootCauseTracerBuilder":
        """Set maximum traversal depth."""
        self._max_depth = depth
        return self
    
    def with_min_confidence(self, confidence: float) -> "RootCauseTracerBuilder":
        """Set minimum root cause confidence."""
        self._min_confidence = confidence
        return self
    
    def build(self) -> RootCauseTracer:
        """Build the RootCauseTracer instance."""
        if self._graph is None:
            raise ValueError("Dependency graph must be provided")
        
        return RootCauseTracer(
            dependency_graph=self._graph,
            anomaly_threshold=self._anomaly_threshold,
            critical_threshold=self._critical_threshold,
            max_depth=self._max_depth,
            min_root_cause_confidence=self._min_confidence
        )
