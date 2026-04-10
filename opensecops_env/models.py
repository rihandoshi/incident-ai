"""
OpenSecOpsEnv Models
====================
Type-safe data structures for the SecOps incident response environment.
Follows the OpenEnv dataclass-based contract.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Action Types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All legal action types an agent can take."""
    QUERY_LOGS = "query_logs"
    INSPECT_METRICS = "inspect_metrics"
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    BLOCK_IP = "block_ip"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    RUN_SECURITY_SCAN = "run_security_scan"
    ISOLATE_SERVICE = "isolate_service"
    SUBMIT_DIAGNOSIS = "submit_diagnosis"


# ---------------------------------------------------------------------------
# OpenEnv Base Stubs (so env runs standalone without openenv-core installed)
# ---------------------------------------------------------------------------

@dataclass
class _BaseAction:
    pass


@dataclass
class _BaseObservation:
    pass


@dataclass
class _BaseState:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0


try:
    from openenv.core.env_server import Action as _OEAction
    from openenv.core.env_server import Observation as _OEObservation
    from openenv.core.env_server import State as _OEState

    _ActionBase = _OEAction
    _ObsBase = _OEObservation
    _StateBase = _OEState
except ImportError:
    _ActionBase = _BaseAction  # type: ignore[assignment]
    _ObsBase = _BaseObservation  # type: ignore[assignment]
    _StateBase = _BaseState  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class SecOpsAction(_ActionBase):  # type: ignore[misc]
    """
    A structured action an agent can take inside the SecOps environment.

    Attributes
    ----------
    action_type : ActionType
        The kind of action to perform.
    parameters : dict
        Action-specific parameters.  Examples:
          - query_logs        → {"service": "auth", "keywords": ["error"]}
          - inspect_metrics   → {"service": "db"}
          - restart_service   → {"service": "cache"}
          - scale_service     → {"service": "api", "replicas": 5}
          - block_ip          → {"ip": "1.2.3.4"}
          - rollback_deployment → {"service": "api", "version": "v1.2"}
          - run_security_scan → {"target": "auth"}
          - isolate_service   → {"service": "infected_svc"}
          - submit_diagnosis  → {"label": "cyber_attack:ddos"}
    """

    action_type: str = ActionType.QUERY_LOGS.value
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise to string value so JSON round-trips safely
        if isinstance(self.action_type, ActionType):
            self.action_type = self.action_type.value


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class ServiceMetrics:
    """Per-service numeric metrics."""
    cpu: float = 0.0
    memory: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0


@dataclass
class SecOpsObservation(_ObsBase):  # type: ignore[misc]
    """
    Partial observation returned to the agent at each step.

    NOTE: The true root cause is *never* directly exposed here.
    The agent must infer it from indirect signals.

    Attributes
    ----------
    alerts : list[dict]
        Triggered monitoring alerts, e.g.
        [{"service": "auth", "type": "high_cpu", "severity": "warning"}]
    metrics : dict[str, dict]
        Per-service metrics snapshot.
        {service_name: {"cpu": …, "memory": …, "latency": …, "error_rate": …}}
    logs : list[str]
        Recent (and potentially noisy) log lines from relevant services.
    topology : dict[str, list[str]]
        Service dependency graph. {service: [downstream_deps]}
    last_action_result : str
        Human-readable outcome of the previous action.
    time_step : int
        Current time step within the episode.
    available_actions : list[str]
        Action types the agent may use (constant, for reference).
    """

    alerts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    topology: dict[str, list[str]] = field(default_factory=dict)
    last_action_result: str = ""
    time_step: int = 0
    available_actions: list[str] = field(
        default_factory=lambda: [a.value for a in ActionType]
    )


# ---------------------------------------------------------------------------
# Hidden State (internal only – never sent to agent directly)
# ---------------------------------------------------------------------------

@dataclass
class HiddenState:
    """
    The ground truth that the agent must discover through investigation.
    This is *not* part of the Observation; it drives environment dynamics.
    """

    true_root_cause: str = ""           # "infra_failure" | "misconfiguration" | "cyber_attack"
    subtype: str = ""                   # e.g. "memory_leak", "ddos", "bad_config"
    affected_services: list[str] = field(default_factory=list)
    attack_progress: float = 0.0        # 0-1; relevant for cyber attacks
    noise_level: float = 0.2            # fraction of misleading signals
    diagnosis_submitted: bool = False
    submitted_label: str = ""


# ---------------------------------------------------------------------------
# Episode State (exposed via state() endpoint)
# ---------------------------------------------------------------------------

@dataclass
class SecOpsState(_StateBase):  # type: ignore[misc]
    """
    Full internal state for debugging / grading.  Includes hidden state.

    The hidden_state dict encodes HiddenState fields as plain JSON-serialisable
    primitives so callers don't need to import HiddenState.
    """

    task_id: str = ""
    max_steps: int = 50
    cumulative_reward: float = 0.0
    done: bool = False
    investigation_actions: list[str] = field(default_factory=list)
    mitigation_actions: list[str] = field(default_factory=list)
    correct_mitigations: list[str] = field(default_factory=list)

    # Ground truth (exposed in state() for graders/debuggers)
    hidden_state: dict[str, Any] = field(default_factory=dict)

    # Live metrics stored internally
    _service_metrics: dict[str, ServiceMetrics] = field(
        default_factory=dict, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
            "max_steps": self.max_steps,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "investigation_actions": self.investigation_actions,
            "mitigation_actions": self.mitigation_actions,
            "correct_mitigations": self.correct_mitigations,
            "hidden_state": self.hidden_state,
        }
