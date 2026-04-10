"""
OpenSecOpsEnv – Core Environment
==================================
Implements reset(), step(), state() with full OpenEnv semantics.

Architecture
------------
- Environment is a pure Python class (no external deps beyond stdlib).
- A FastAPI wrapper (server/app.py) exposes it over HTTP.
- Can be used directly in unit tests / inference scripts without Docker.
"""

from __future__ import annotations

import copy
import random
import re
import uuid
from typing import Any

from opensecops_env.models import (
    ActionType,
    HiddenState,
    SecOpsAction,
    SecOpsObservation,
    SecOpsState,
    ServiceMetrics,
)
from opensecops_env.tasks.task_definitions import TaskConfig, get_task

try:
    from openenv.core.env_server import Environment
    _EnvironmentBase = Environment
except ImportError:
    class _EnvironmentBase:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Investigation & Mitigation action sets (for reward shaping)
# ---------------------------------------------------------------------------

INVESTIGATION_ACTIONS = {
    ActionType.QUERY_LOGS.value,
    ActionType.INSPECT_METRICS.value,
    ActionType.RUN_SECURITY_SCAN.value,
}

MITIGATION_ACTIONS = {
    ActionType.RESTART_SERVICE.value,
    ActionType.SCALE_SERVICE.value,
    ActionType.BLOCK_IP.value,
    ActionType.ROLLBACK_DEPLOYMENT.value,
    ActionType.ISOLATE_SERVICE.value,
}


# ---------------------------------------------------------------------------
# Per-scenario dynamics helpers
# ---------------------------------------------------------------------------

def _drift_metrics(
    metrics: dict[str, ServiceMetrics],
    hidden: HiddenState,
    rng: random.Random,
    noise_level: float,
) -> None:
    """Mutate service metrics in-place to simulate time evolution."""

    rc = hidden.true_root_cause
    sub = hidden.subtype
    ap = hidden.attack_progress

    for svc, m in metrics.items():
        # Baseline brownian drift (small noise)
        m.cpu       += rng.gauss(0, noise_level * 2)
        m.memory    += rng.gauss(0, noise_level * 1.5)
        m.latency   += rng.gauss(0, noise_level * 5)
        m.error_rate = max(0.0, m.error_rate + rng.gauss(0, noise_level * 0.5))

        if rc == "infra_failure" and sub == "memory_leak":
            if svc in hidden.affected_services:
                m.memory    = min(100.0, m.memory + 1.5)
                m.latency   = max(0.0, m.latency + 10.0)
                m.error_rate = min(50.0, m.error_rate + 0.3)

        elif rc == "cyber_attack" and sub == "ddos":
            if svc in hidden.affected_services:
                spike = ap * 5.0
                m.cpu    = min(100.0, m.cpu    + spike + rng.gauss(0, 2))
                m.latency = min(5000.0, m.latency + spike * 20 + rng.gauss(0, 10))
                m.error_rate = min(100.0, m.error_rate + spike * 2)

        elif rc == "cyber_attack" and sub == "data_exfiltration":
            if svc in hidden.affected_services:
                m.cpu    = min(100.0, m.cpu    + ap * 1.5)
                m.memory = min(100.0, m.memory + ap * 0.8)

        # Clamp
        m.cpu       = max(0.0, min(100.0, m.cpu))
        m.memory    = max(0.0, min(100.0, m.memory))
        m.latency   = max(0.0, m.latency)
        m.error_rate = max(0.0, min(100.0, m.error_rate))


def _generate_logs(
    hidden: HiddenState,
    rng: random.Random,
    task_initial_logs: list[str],
    step: int,
) -> list[str]:
    """Return a fresh slice of logs with some randomness."""

    # Base pool from task definition
    pool = list(task_initial_logs)

    rc = hidden.true_root_cause
    sub = hidden.subtype
    ap = hidden.attack_progress

    if rc == "infra_failure" and sub == "memory_leak":
        pool += [
            f"[auth] WARN  Heap dump triggered (step {step}): 89% used",
            "[auth] ERROR Java heap space – GC overhead limit exceeded",
        ]

    elif rc == "cyber_attack" and sub == "ddos":
        if ap > 0.5:
            pool += [
                "[gateway] CRIT  SYN flood detected from /24 block 203.0.113.0",
                "[api]  ERROR  Worker pool exhausted – dropping requests",
            ]

    elif rc == "cyber_attack" and sub == "data_exfiltration":
        if ap > 0.4:
            pool += [
                "[db]   CRIT  Unusual 8 GB export to external host 10.0.0.99",
                "[audit] CRIT  Mass data access by reports_bot – exfiltration risk",
            ]

    # Inject misleading noise
    noise_logs = [
        "[syslog] INFO  NTP sync OK",
        "[kernel] INFO  Disk I/O stable",
        "[cron]   INFO  Backup job completed successfully",
        "[syslog] WARN  Entropy pool low – harmless",
        "[kernel] DEBUG CPU throttling event (thermal) – minor",
    ]
    n_noise = int(len(pool) * hidden.noise_level)
    pool += rng.sample(noise_logs, min(n_noise, len(noise_logs)))

    rng.shuffle(pool)
    # Return only the most recent 8 lines (partial window)
    return pool[:8]


def _evolve_attack(hidden: HiddenState, isolated: set[str]) -> None:
    """Advance attack progress; slow it down if affected services isolated."""
    if hidden.true_root_cause != "cyber_attack":
        return
    slowdown = 0.5 if any(s in isolated for s in hidden.affected_services) else 1.0
    hidden.attack_progress = min(1.0, hidden.attack_progress + 0.08 * slowdown)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OpenSecOpsEnv(_EnvironmentBase):  # type: ignore[misc]
    """
    OpenSecOpsEnv – SecOps Incident Response environment.

    Usage
    -----
    >>> env = OpenSecOpsEnv()
    >>> obs = env.reset("easy_memory_leak")
    >>> result = env.step(SecOpsAction(action_type="query_logs",
    ...                                parameters={"service": "auth"}))
    >>> obs, reward, done, info = result
    """

    MAX_STEPS_DEFAULT = 50

    def __init__(self) -> None:
        # Try to call super().__init__() for openenv-core compat
        try:
            super().__init__()
        except Exception:
            pass
        self._state = SecOpsState()
        self._hidden = HiddenState()
        self._metrics: dict[str, ServiceMetrics] = {}
        self._rng = random.Random(42)
        self._task_cfg: TaskConfig = {}
        self._isolated_services: set[str] = set()
        self._blocked_ips: set[str] = set()
        self._restarted_services: set[str] = set()
        self._scaled_services: set[str] = set()
        self._rolled_back: set[str] = set()
        self._scanned: set[str] = set()
        # Diminishing returns: tracks how many times each service was investigated
        self._investigated: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy_memory_leak") -> SecOpsObservation:
        """
        Initialise a fresh episode for the given task.

        Parameters
        ----------
        task_id : str
            One of "easy_memory_leak", "medium_ddos_cascade",
            "hard_data_exfiltration".

        Returns
        -------
        SecOpsObservation
            Initial partial observation.
        """
        cfg = get_task(task_id)
        self._task_cfg = cfg

        # Seed for strict reproducibility
        self._rng = random.Random(cfg["seed"])

        # Reset transient sets
        self._isolated_services = set()
        self._blocked_ips = set()
        self._restarted_services = set()
        self._scaled_services = set()
        self._rolled_back = set()
        self._scanned = set()
        self._investigated = {}

        # Build hidden state
        self._hidden = HiddenState(
            true_root_cause=cfg["true_root_cause"],
            subtype=cfg["subtype"],
            affected_services=list(cfg["affected_services"]),
            attack_progress=cfg.get("attack_progress_start", 0.0),
            noise_level=cfg["noise_level"],
        )

        # Build metrics from task config
        self._metrics = {
            svc: ServiceMetrics(**vals)
            for svc, vals in cfg["initial_metrics"].items()
        }

        # Build episode state
        self._state = SecOpsState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=cfg["max_steps"],
            cumulative_reward=0.0,
            done=False,
            hidden_state=self._hidden_as_dict(),
            correct_mitigations=list(cfg["correct_mitigations"]),
        )

        return self._build_observation("Episode started. Investigate the system.")

    def step(
        self, action: SecOpsAction
    ) -> tuple[SecOpsObservation, float, bool, dict[str, Any]]:
        """
        Execute an action and advance the environment by one step.

        Parameters
        ----------
        action : SecOpsAction

        Returns
        -------
        (observation, reward, done, info)
        """
        if self._state.done:
            obs = self._build_observation("Episode already finished.")
            return obs, 0.0, True, {"warning": "already_done"}

        self._state.step_count += 1
        reward, result_msg, done = self._process_action(action)

        # Evolve environment dynamics
        _drift_metrics(self._metrics, self._hidden, self._rng, self._hidden.noise_level)
        _evolve_attack(self._hidden, self._isolated_services)

        # Check episode termination
        if self._state.step_count >= self._state.max_steps:
            done = True
            result_msg += " [Max steps reached]"

        self._state.done = done
        self._state.cumulative_reward += reward
        self._state.hidden_state = self._hidden_as_dict()

        obs = self._build_observation(result_msg)
        info = {
            "step": self._state.step_count,
            "cumulative_reward": self._state.cumulative_reward,
            "done": done,
        }
        return obs, round(reward, 4), done, info

    @property
    def state(self) -> SecOpsState:
        """Full internal state (for debugging / grading)."""
        self._state.hidden_state = self._hidden_as_dict()
        return self._state

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def _process_action(
        self, action: SecOpsAction
    ) -> tuple[float, str, bool]:
        """Dispatch action, compute reward, return (reward, message, done)."""

        at = action.action_type
        params = action.parameters or {}

        # ---- Investigation actions ----
        if at == ActionType.QUERY_LOGS.value:
            return self._act_query_logs(params)

        if at == ActionType.INSPECT_METRICS.value:
            return self._act_inspect_metrics(params)

        if at == ActionType.RUN_SECURITY_SCAN.value:
            return self._act_run_security_scan(params)

        # ---- Mitigation actions ----
        if at == ActionType.RESTART_SERVICE.value:
            return self._act_restart_service(params)

        if at == ActionType.SCALE_SERVICE.value:
            return self._act_scale_service(params)

        if at == ActionType.BLOCK_IP.value:
            return self._act_block_ip(params)

        if at == ActionType.ROLLBACK_DEPLOYMENT.value:
            return self._act_rollback_deployment(params)

        if at == ActionType.ISOLATE_SERVICE.value:
            return self._act_isolate_service(params)

        # ---- Terminal action ----
        if at == ActionType.SUBMIT_DIAGNOSIS.value:
            return self._act_submit_diagnosis(params)

        return -0.2, f"Unknown action type '{at}'.", False

    # --- Query Logs ---
    def _act_query_logs(self, params: dict) -> tuple[float, str, bool]:
        svc = params.get("service", "")
        if svc not in self._metrics:
            return -0.2, f"Service '{svc}' not found.", False

        self._state.investigation_actions.append(f"query_logs:{svc}")
        logs = _generate_logs(
            self._hidden, self._rng,
            self._task_cfg.get("initial_logs", []),
            self._state.step_count,
        )
        # Filter to logs mentioning the queried service
        relevant = [l for l in logs if f"[{svc}]" in l or svc in l.lower()]
        if not relevant:
            relevant = [f"[{svc}] INFO  No new events."]

        # Reward: useful if looking at an affected service (diminishing returns)
        if svc in self._hidden.affected_services:
            times = self._investigated.get(svc, 0)
            self._investigated[svc] = times + 1
            reward = 0.2 if times == 0 else (0.05 if times == 1 else 0.0)
        else:
            reward = -0.05   # mild penalty for investigating healthy service
        return reward, f"Logs for '{svc}':\n" + "\n".join(relevant[:5]), False

    # --- Inspect Metrics ---
    def _act_inspect_metrics(self, params: dict) -> tuple[float, str, bool]:
        svc = params.get("service", "")
        if svc and svc not in self._metrics:
            return -0.2, f"Service '{svc}' not found.", False

        self._state.investigation_actions.append(
            f"inspect_metrics:{svc or 'all'}"
        )
        if svc:
            m = self._metrics[svc]
            msg = (
                f"Metrics for {svc}: cpu={m.cpu:.1f}% mem={m.memory:.1f}% "
                f"latency={m.latency:.1f}ms err={m.error_rate:.2f}%"
            )
            # Diminishing returns for repeated inspection of same service
            if svc in self._hidden.affected_services:
                times = self._investigated.get(svc, 0)
                self._investigated[svc] = times + 1
                reward = 0.2 if times == 0 else (0.05 if times == 1 else 0.0)
            else:
                reward = -0.05
        else:
            # Global inspect_metrics({}) — useful once, neutral after that
            times = self._investigated.get("__all__", 0)
            self._investigated["__all__"] = times + 1
            lines = [
                f"  {s}: cpu={m.cpu:.1f}% mem={m.memory:.1f}% "
                f"latency={m.latency:.1f}ms err={m.error_rate:.2f}%"
                for s, m in self._metrics.items()
            ]
            msg = "All metrics:\n" + "\n".join(lines)
            reward = 0.2 if times == 0 else 0.0

        return reward, msg, False

    # --- Security Scan ---
    def _act_run_security_scan(self, params: dict) -> tuple[float, str, bool]:
        target = params.get("target", "")
        if target and target not in self._metrics:
            return -0.2, f"Target '{target}' not found.", False

        self._scanned.add(target)
        self._state.investigation_actions.append(f"run_security_scan:{target}")

        rc = self._hidden.true_root_cause
        sub = self._hidden.subtype

        if rc == "cyber_attack" and target in self._hidden.affected_services:
            if sub == "data_exfiltration":
                msg = (
                    f"SECURITY SCAN – {target}: "
                    "ALERT: Unusual outbound data transfer detected. "
                    "Suspicious process 'reports_bot' opened 847 connections. "
                    "Recommend isolation."
                )
            elif sub == "ddos":
                msg = (
                    f"SECURITY SCAN – {target}: "
                    "ALERT: Connection table 98% full – SYN flood indicators. "
                    "Recommend blocking attacking IP ranges."
                )
            else:
                msg = f"SECURITY SCAN – {target}: Anomalies detected."
            # Diminishing returns: +0.3 first scan, 0.0 on repeat
            already = f"run_security_scan:{target}" in (
                self._state.investigation_actions[:-1]  # exclude just-appended entry
            )
            reward = 0.0 if already else 0.3
        else:
            msg = f"SECURITY SCAN – {target}: No critical vulnerabilities found."
            reward = -0.05
        return reward, msg, False

    # --- Restart Service ---
    def _act_restart_service(self, params: dict) -> tuple[float, str, bool]:
        svc = params.get("service", "")
        if svc not in self._metrics:
            return -0.2, f"Service '{svc}' not found.", False

        self._restarted_services.add(svc)
        key = f"restart_service:{svc}"

        if key in self._task_cfg.get("correct_mitigations", []):
            # Correct mitigation: reset metrics for affected service
            self._metrics[svc].memory    = 35.0
            self._metrics[svc].latency   = 80.0
            self._metrics[svc].error_rate = 0.5
            self._state.mitigation_actions.append(key)
            reward = 0.5
            msg = f"Service '{svc}' restarted successfully. Metrics normalising."
        elif svc not in self._hidden.affected_services:
            # Restarting wrong service – harmful
            m = self._metrics[svc]
            m.error_rate = min(100.0, m.error_rate + 5.0)
            m.latency    = min(5000.0, m.latency + 200.0)
            reward = -0.5
            msg = (
                f"Service '{svc}' restarted unnecessarily. "
                "Caused brief outage – error_rate increased."
            )
        else:
            # Affected service but wrong action for root cause
            reward = -0.2
            msg = (
                f"Restarting '{svc}' had minimal effect – "
                "root cause requires a different mitigation."
            )
        return reward, msg, False

    # --- Scale Service ---
    def _act_scale_service(self, params: dict) -> tuple[float, str, bool]:
        svc = params.get("service", "")
        replicas = params.get("replicas", 3)
        if svc not in self._metrics:
            return -0.2, f"Service '{svc}' not found.", False

        self._scaled_services.add(svc)
        key = f"scale_service:{svc}"

        if key in self._task_cfg.get("correct_mitigations", []):
            self._metrics[svc].cpu = max(0.0, self._metrics[svc].cpu - 20.0)
            self._metrics[svc].latency = max(0.0, self._metrics[svc].latency - 200.0)
            self._state.mitigation_actions.append(key)
            reward = 0.5
            msg = f"Scaled '{svc}' to {replicas} replicas. Load distributed."
        else:
            reward = -0.1
            msg = f"Scaling '{svc}' had no significant effect on the root cause."
        return reward, msg, False

    # --- Block IP ---
    def _act_block_ip(self, params: dict) -> tuple[float, str, bool]:
        ip = params.get("ip", "")
        if not ip:
            return -0.2, "No IP provided.", False

        self._blocked_ips.add(ip)
        key = f"block_ip:{ip}"

        correct_blocks = [
            m for m in self._task_cfg.get("correct_mitigations", [])
            if m.startswith("block_ip:")
        ]

        if key in correct_blocks:
            # Reduce attack-related metrics
            for svc in self._hidden.affected_services:
                if svc in self._metrics:
                    self._metrics[svc].cpu = max(0.0, self._metrics[svc].cpu - 15.0)
                    self._metrics[svc].latency = max(
                        0.0, self._metrics[svc].latency - 200.0
                    )
            self._state.mitigation_actions.append(key)
            self._hidden.attack_progress = max(
                0.0, self._hidden.attack_progress - 0.2
            )
            reward = 0.5
            msg = f"IP {ip} blocked. Traffic from that host dropped."
        else:
            # Blocking legit IP → increases error rate
            for svc in self._metrics.values():
                svc.error_rate = min(100.0, svc.error_rate + 2.0)
            reward = -0.5
            msg = (
                f"IP {ip} blocked, but it was a legitimate host! "
                "Error rates across services increased."
            )
        return reward, msg, False

    # --- Rollback Deployment ---
    def _act_rollback_deployment(self, params: dict) -> tuple[float, str, bool]:
        svc = params.get("service", "")
        version = params.get("version", "previous")
        if svc not in self._metrics:
            return -0.2, f"Service '{svc}' not found.", False

        key = f"rollback_deployment:{svc}"

        if key in self._task_cfg.get("correct_mitigations", []):
            # Only grant reward first time this rollback is performed
            if key not in self._state.mitigation_actions:
                # Normalise metrics — bad config is gone after rollback
                self._metrics[svc].latency    = max(80.0, self._metrics[svc].latency * 0.3)
                self._metrics[svc].error_rate = max(0.5,  self._metrics[svc].error_rate * 0.1)
                self._metrics[svc].cpu        = max(0.0,  self._metrics[svc].cpu - 20.0)
                self._state.mitigation_actions.append(key)
                self._rolled_back.add(svc)
                reward = 0.5
                msg = f"Deployment of '{svc}' rolled back to {version}. Latency and error rate normalising."
            else:
                reward = 0.0
                msg = f"'{svc}' was already rolled back."
        else:
            reward = -0.1
            msg = f"Rolling back '{svc}' had no meaningful impact on the incident."
        return reward, msg, False

    # --- Isolate Service ---
    def _act_isolate_service(self, params: dict) -> tuple[float, str, bool]:
        svc = params.get("service", "")
        if svc not in self._metrics:
            return -0.2, f"Service '{svc}' not found.", False

        self._isolated_services.add(svc)
        key = f"isolate_service:{svc}"

        if key in self._task_cfg.get("correct_mitigations", []):
            self._hidden.attack_progress = max(
                0.0, self._hidden.attack_progress - 0.35
            )
            self._state.mitigation_actions.append(key)
            reward = 0.5
            msg = (
                f"Service '{svc}' isolated from the network. "
                "Attack vector contained – attack progress reduced."
            )
        elif svc not in self._hidden.affected_services:
            # Isolating healthy service → user-facing impact
            m = self._metrics[svc]
            m.error_rate = min(100.0, m.error_rate + 8.0)
            reward = -0.5
            msg = (
                f"Service '{svc}' isolated unnecessarily. "
                "Users are now unable to reach that service!"
            )
        else:
            reward = -0.1
            msg = f"Isolating '{svc}' didn't fully address the root cause."
        return reward, msg, False

    # --- Submit Diagnosis ---
    def _act_submit_diagnosis(self, params: dict) -> tuple[float, str, bool]:
        label = params.get("label", "")
        correct = self._task_cfg.get("correct_label", "")
        self._hidden.diagnosis_submitted = True
        self._hidden.submitted_label = label

        if label == correct:
            reward = 1.0
            msg = f"✅ Correct diagnosis: '{label}'. Episode complete."
        else:
            reward = -1.0
            msg = (
                f"❌ Incorrect diagnosis: '{label}'. "
                f"The true root cause was '{correct}'."
            )
        return reward, msg, True   # always terminates episode

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, last_action_result: str) -> SecOpsObservation:
        # Snapshot metrics
        metrics_snapshot = {
            svc: {
                "cpu": round(m.cpu, 2),
                "memory": round(m.memory, 2),
                "latency": round(m.latency, 2),
                "error_rate": round(m.error_rate, 2),
            }
            for svc, m in self._metrics.items()
        }

        # Dynamic alerts based on current metrics
        alerts = self._generate_alerts()

        # Logs (partial window)
        logs = _generate_logs(
            self._hidden,
            self._rng,
            self._task_cfg.get("initial_logs", []),
            self._state.step_count,
        )

        return SecOpsObservation(
            alerts=alerts,
            metrics=metrics_snapshot,
            logs=logs,
            topology=self._task_cfg.get("topology", {}),
            last_action_result=last_action_result,
            time_step=self._state.step_count,
        )

    def _generate_alerts(self) -> list[dict[str, Any]]:
        """Generate threshold-based alerts from current metrics."""
        alerts: list[dict[str, Any]] = []

        for svc, m in self._metrics.items():
            if m.cpu > 90:
                alerts.append({"service": svc, "type": "high_cpu",
                                "severity": "critical",
                                "message": f"{svc} CPU {m.cpu:.1f}%"})
            elif m.cpu > 75:
                alerts.append({"service": svc, "type": "high_cpu",
                                "severity": "warning",
                                "message": f"{svc} CPU {m.cpu:.1f}%"})
            if m.memory > 85:
                alerts.append({"service": svc, "type": "high_memory",
                                "severity": "critical",
                                "message": f"{svc} memory {m.memory:.1f}%"})
            elif m.memory > 75:
                alerts.append({"service": svc, "type": "high_memory",
                                "severity": "warning",
                                "message": f"{svc} memory {m.memory:.1f}%"})
            if m.latency > 500:
                alerts.append({"service": svc, "type": "high_latency",
                                "severity": "critical",
                                "message": f"{svc} latency {m.latency:.0f}ms"})
            elif m.latency > 200:
                alerts.append({"service": svc, "type": "high_latency",
                                "severity": "warning",
                                "message": f"{svc} latency {m.latency:.0f}ms"})
            if m.error_rate > 10:
                alerts.append({"service": svc, "type": "high_error_rate",
                                "severity": "critical",
                                "message": f"{svc} error_rate {m.error_rate:.1f}%"})
            elif m.error_rate > 5:
                alerts.append({"service": svc, "type": "high_error_rate",
                                "severity": "warning",
                                "message": f"{svc} error_rate {m.error_rate:.1f}%"})

        # Inject static misleading alerts from task config
        static_alerts = self._task_cfg.get("initial_alerts", [])
        noise = self._hidden.noise_level
        for a in static_alerts:
            if self._rng.random() < noise and a not in alerts:
                alerts.append(a)

        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hidden_as_dict(self) -> dict[str, Any]:
        h = self._hidden
        return {
            "true_root_cause": h.true_root_cause,
            "subtype": h.subtype,
            "affected_services": h.affected_services,
            "attack_progress": round(h.attack_progress, 4),
            "noise_level": h.noise_level,
            "diagnosis_submitted": h.diagnosis_submitted,
            "submitted_label": h.submitted_label,
            "correct_label": self._task_cfg.get("correct_label", ""),
        }
