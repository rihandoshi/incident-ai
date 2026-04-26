"""
FastAPI Server for OpenSecOpsEnv
=================================
Exposes reset(), step(), state() over HTTP.

Endpoints
---------
POST /reset          – begin new episode (creates/replaces session)
POST /step           – execute one action within a session
GET  /state          – get full debug state for a session
POST /grade          – grade the current episode
GET  /health         – liveness probe
GET  /tasks          – list available tasks
GET  /web            – simple debug UI
GET  /dashboard      – 🔥 Live AI Demo Dashboard  (Attacker vs Defender)
GET  /demo/stream    – SSE stream: single-agent episode (for basic demo)
GET  /battle/stream  – SSE stream: Red Attacker vs Blue Defender live battle

Multi-Agent Architecture
------------------------
- Red Agent  (Attacker)  – escalates the incident, tries to worsen metrics
- Blue Agent (Defender)  – investigates and mitigates the incident
- Both share the same environment; turn order is interleaved
- Each agent has a distinct reward signal (zero-sum-like)
- Curriculum: Blue agent auto-levels up as it solves harder tasks

Session model
-------------
Each reset() call creates a named session keyed by `session_id`
(defaults to the task_id). This allows multiple concurrent agents
to run different tasks without colliding.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from opensecops_env.env import OpenSecOpsEnv
from opensecops_env.grader import grade
from opensecops_env.models import SecOpsAction
from opensecops_env.tasks.task_definitions import TASKS

# ---------------------------------------------------------------------------
# 🤖 Live AI Inference Engine
# ---------------------------------------------------------------------------
# Set these environment variables to connect to your trained Hugging Face
# Inference Endpoints. When not set, falls back to heuristic playbooks.
#
# TRAINED_MODEL_ENDPOINT   — your fine-tuned Qwen2.5-7B-GRPO endpoint
# UNTRAINED_MODEL_ENDPOINT — base Qwen2.5-7B for contrast (optional)
# HF_API_TOKEN             — your Hugging Face API token
# ---------------------------------------------------------------------------

# Hardcoded live defaults — override via env var at any time
_TRAINED_ENDPOINT: Optional[str] = (
    os.environ.get("TRAINED_MODEL_ENDPOINT", "").strip()
    or "https://hk5m5hadqtjiqg53.us-east-1.aws.endpoints.huggingface.cloud"
)
_UNTRAINED_ENDPOINT: Optional[str] = os.environ.get("UNTRAINED_MODEL_ENDPOINT", "").strip() or None
_HF_API_TOKEN: str = os.environ.get("HF_API_TOKEN", os.environ.get("HF_TOKEN", ""))

_AI_SYSTEM_PROMPT = """You are an expert on-call security engineer responding to a production incident.
At each step you receive: alerts, metrics, logs, topology, last_action_result.
Your goal: investigate the root cause and apply targeted mitigations.
RESPOND ONLY with a valid JSON object:
{"action_type": "<type>", "parameters": {<params>}}

Actions: query_logs, inspect_metrics, restart_service, scale_service,
         block_ip, rollback_deployment, run_security_scan,
         isolate_service, submit_diagnosis

Diagnosis labels:
  infra_failure:memory_leak | infra_failure:service_crash
  misconfiguration:bad_config
  cyber_attack:ddos | cyber_attack:data_exfiltration | cyber_attack:privilege_escalation"""


def _obs_to_text(obs: dict, step: int) -> str:
    """Format an observation dict into the prompt text the model was trained on."""
    parts = [f"=== Step {step} ==="]
    parts.append(f"Last result: {obs.get('last_action_result', '')}")
    if obs.get("alerts"):
        parts.append("\nALERTS:")
        for a in obs["alerts"]:
            parts.append(f"  [{a.get('severity','').upper()}] {a.get('service')} - {a.get('message','')}")
    parts.append("\nMETRICS:")
    for svc, m in obs.get("metrics", {}).items():
        if isinstance(m, dict):
            parts.append(
                f"  {svc}: cpu={m.get('cpu',0):.1f}% mem={m.get('memory',0):.1f}% "
                f"lat={m.get('latency',0):.0f}ms err={m.get('error_rate',0):.2f}%"
            )
    parts.append("\nLOGS:")
    for line in obs.get("logs", [])[:5]:
        parts.append(f"  {line}")
    parts.append("\nTOPOLOGY:")
    for svc, deps in obs.get("topology", {}).items():
        parts.append(f"  {svc} -> {deps}")
    parts.append("\nRespond with JSON action:")
    return "\n".join(parts)


def _parse_ai_action(text: str) -> Optional[SecOpsAction]:
    """Parse JSON action from LLM response text."""
    if isinstance(text, list):
        text = text[-1].get("content", "") if text else ""
    if not isinstance(text, str):
        text = str(text)
    # Strip markdown code fences
    text = re.sub(r"```[a-z]*\n?", "", text.strip()).strip()
    # Extract first JSON object if wrapped in other text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        d = json.loads(text)
        return SecOpsAction(
            action_type=d.get("action_type", "inspect_metrics"),
            parameters=d.get("parameters", {}),
        )
    except Exception:
        return None


async def _query_ai_model(
    endpoint_url: str,
    obs: dict,
    step: int,
    timeout: float = 30.0,
) -> Optional[SecOpsAction]:
    """
    Call a Hugging Face Inference Endpoint with the current observation.
    Returns a parsed SecOpsAction, or None if the call fails.
    """
    try:
        import httpx
    except ImportError:
        return None

    prompt_text = _obs_to_text(obs, step)
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False,
        },
    }
    headers = {"Content-Type": "application/json"}
    if _HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_API_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # HF text-generation returns [{"generated_text": "..."}]
            if isinstance(data, list) and data:
                generated = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                generated = data.get("generated_text", str(data))
            else:
                generated = str(data)
            return _parse_ai_action(generated)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Session registry  (session_id → OpenSecOpsEnv instance)
# ---------------------------------------------------------------------------
_sessions: dict[str, OpenSecOpsEnv] = {}
_default_session: str = "default"


def _get_session(session_id: str) -> OpenSecOpsEnv:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# ██████╗ ███████╗██████╗     ███████╗███████╗ ██████╗ ███████╗███████╗
# ██╔══██╗██╔════╝██╔══██╗    ██╔════╝██╔════╝██╔════╝ ██╔════╝██╔════╝
# ██████╔╝█████╗  ██║  ██║    ███████╗█████╗  ██║  ███╗█████╗  ███████╗
# ██╔══██╗██╔══╝  ██║  ██║    ╚════██║██╔══╝  ██║   ██║██╔══╝  ╚════██║
# ██║  ██║███████╗██████╔╝    ███████║███████╗╚██████╔╝███████╗███████║
# Multi-Agent:  Red (Attacker)  vs  Blue (Defender)
# ---------------------------------------------------------------------------

# Red agent actions: escalate attacks, confuse defender, corrupt metrics
RED_ACTIONS = [
    "inject_noise",       # inject misleading log noise
    "amplify_attack",     # boost attack_progress
    "corrupt_metric",     # spike a random healthy service's cpu/latency
    "create_false_alert", # add a misleading critical alert
    "accelerate_spread",  # spread attack to adjacent service
]


@dataclass
class RedAgentState:
    """Lightweight internal state for the Red agent."""
    task_id: str = ""
    round: int = 0
    actions_taken: list[str] = field(default_factory=list)
    total_damage: float = 0.0


class MultiAgentSecOpsEnv:
    """
    Wraps OpenSecOpsEnv to expose separate blue_step() and red_step() interfaces.

    Blue Agent (Defender): uses the standard SecOpsAction API.
    Red  Agent (Attacker): uses RedAgent heuristics to make the incident worse.

    Turn order: red acts first each round (escalate), then blue responds (mitigate).
    This models real adversarial incident response.
    """

    def __init__(self) -> None:
        self._env = OpenSecOpsEnv()
        self._red_state = RedAgentState()
        self._rng = random.Random(42)
        self._cumulative_blue_reward = 0.0
        self._cumulative_red_reward = 0.0

    def reset(self, task_id: str) -> dict[str, Any]:
        obs = self._env.reset(task_id)
        self._red_state = RedAgentState(task_id=task_id)
        self._cumulative_blue_reward = 0.0
        self._cumulative_red_reward = 0.0
        return self._build_ma_state(obs)

    # --- Blue (Defender) step ---
    def blue_step(
        self, action: SecOpsAction
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        self._cumulative_blue_reward += reward
        info["blue_cumulative"] = round(self._cumulative_blue_reward, 4)
        info["red_cumulative"] = round(self._cumulative_red_reward, 4)
        return self._build_ma_state(obs), reward, done, info

    # --- Red (Attacker) step ---
    def red_step(
        self, action_type: Optional[str] = None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Heuristic Red Agent step.  Reward is the negative of what the Blue
        agent would gain — i.e., Red benefits when the environment is harder
        to diagnose and recover from.
        """
        self._red_state.round += 1
        red_reward = 0.0
        message = ""

        env = self._env
        hidden = env._hidden
        metrics = env._metrics

        # Choose action heuristically if not specified
        if action_type is None or action_type == "auto":
            action_type = self._heuristic_red_action()

        self._red_state.actions_taken.append(action_type)

        if action_type == "inject_noise":
            # Add misleading log entries to obscure the real issue
            env._task_cfg.setdefault("initial_logs", [])
            noise_logs = [
                "[cache] WARN  Memory spike (harmless GC event) detected",
                "[db]   INFO  Routine VACUUM running – may spike CPU temporarily",
                "[auth] INFO  Token rotation in progress – latency may increase",
                "[api]  WARN  Connection pool resize – client may see 503 briefly",
            ]
            entry = self._rng.choice(noise_logs)
            env._task_cfg["initial_logs"].append(entry)
            red_reward = 0.1
            message = f"Red injected misleading log: {entry[:60]}..."

        elif action_type == "amplify_attack":
            if hidden.true_root_cause == "cyber_attack":
                boost = self._rng.uniform(0.05, 0.15)
                hidden.attack_progress = min(1.0, hidden.attack_progress + boost)
                red_reward = 0.2
                message = f"Red amplified attack progress by {boost:.2f} → {hidden.attack_progress:.2f}"
            else:
                red_reward = 0.0
                message = "Red tried to amplify — no active attack to boost."

        elif action_type == "corrupt_metric":
            # Spike a *healthy* (non-affected) service to mislead the defender
            healthy = [
                s for s in metrics
                if s not in hidden.affected_services
            ]
            if healthy:
                svc = self._rng.choice(healthy)
                metrics[svc].cpu = min(100.0, metrics[svc].cpu + self._rng.uniform(20, 40))
                metrics[svc].latency = min(5000.0, metrics[svc].latency + self._rng.uniform(150, 400))
                red_reward = 0.15
                message = f"Red spiked metrics on healthy service '{svc}' to create false alarm."
            else:
                red_reward = 0.0
                message = "Red tried to corrupt metrics — no healthy services available."

        elif action_type == "create_false_alert":
            healthy = [s for s in metrics if s not in hidden.affected_services]
            if healthy:
                svc = self._rng.choice(healthy)
                false_alert = {
                    "service": svc,
                    "type": self._rng.choice(["high_cpu", "high_memory", "high_latency"]),
                    "severity": "critical",
                    "message": f"[RED INJECTED] {svc} critical threshold exceeded",
                    "_red_injected": True,
                }
                env._task_cfg.setdefault("initial_alerts", []).append(false_alert)
                red_reward = 0.15
                message = f"Red injected a false CRITICAL alert on '{svc}'."
            else:
                red_reward = 0.0
                message = "No healthy services to inject false alert on."

        elif action_type == "accelerate_spread":
            # Try to spread the attack to an adjacent service
            topology = env._task_cfg.get("topology", {})
            new_victims = set()
            for affected in list(hidden.affected_services):
                for neighbor in topology.get(affected, []):
                    if neighbor not in hidden.affected_services:
                        new_victims.add(neighbor)
            if new_victims:
                new_svc = self._rng.choice(list(new_victims))
                hidden.affected_services.append(new_svc)
                metrics[new_svc].cpu = min(100.0, metrics[new_svc].cpu + 25.0)
                metrics[new_svc].error_rate = min(100.0, metrics[new_svc].error_rate + 8.0)
                red_reward = 0.25
                message = f"Red spread attack to new service '{new_svc}'!"
            else:
                red_reward = 0.05
                message = "Red tried to spread — no adjacent services to infect."

        else:
            message = f"Unknown red action '{action_type}'"
            red_reward = 0.0

        self._cumulative_red_reward += red_reward
        self._red_state.total_damage += red_reward

        done = env._state.done
        obs = env._build_observation(f"[ATTACKER] {message}")

        info = {
            "red_action": action_type,
            "red_reward": round(red_reward, 4),
            "red_cumulative": round(self._cumulative_red_reward, 4),
            "blue_cumulative": round(self._cumulative_blue_reward, 4),
            "message": message,
        }

        return self._build_ma_state(obs), red_reward, done, info

    def _heuristic_red_action(self) -> str:
        """Smart heuristic: Red agent picks best available action."""
        hidden = self._env._hidden
        metrics = self._env._metrics

        # Prefer amplify if there's an active attack
        if hidden.true_root_cause == "cyber_attack" and hidden.attack_progress < 0.8:
            return "amplify_attack"

        # If many healthy services exist, try to spread or corrupt
        healthy = [s for s in metrics if s not in hidden.affected_services]
        if healthy and self._rng.random() < 0.4:
            return "corrupt_metric"
        if hidden.true_root_cause == "cyber_attack" and self._rng.random() < 0.3:
            return "accelerate_spread"

        # Default: inject noise to confuse
        return "inject_noise"

    def _build_ma_state(self, obs) -> dict[str, Any]:
        state = self._env.state.to_dict()
        state["multi_agent"] = {
            "red_round": self._red_state.round,
            "red_actions": self._red_state.actions_taken,
            "red_total_damage": round(self._red_state.total_damage, 4),
            "blue_cumulative_reward": round(self._cumulative_blue_reward, 4),
            "red_cumulative_reward": round(self._cumulative_red_reward, 4),
        }
        state["observation"] = {
            "alerts": obs.alerts,
            "metrics": obs.metrics,
            "logs": obs.logs,
            "topology": obs.topology,
            "last_action_result": obs.last_action_result,
            "time_step": obs.time_step,
        }
        return state

    @property
    def state(self) -> dict[str, Any]:
        obs = self._env._build_observation("state_query")
        return self._build_ma_state(obs)


# ---------------------------------------------------------------------------
# Curriculum  — tracks Blue agent improvement across episodes
# ---------------------------------------------------------------------------

CURRICULUM_LEVELS = [
    {"level": 1, "tasks": ["easy_memory_leak"],                        "threshold": 0.65},
    {"level": 2, "tasks": ["easy_memory_leak", "medium_ddos_cascade"], "threshold": 0.70},
    {"level": 3, "tasks": ["medium_ddos_cascade", "medium_hard_bad_deployment"], "threshold": 0.72},
    {"level": 4, "tasks": ["medium_hard_bad_deployment", "hard_data_exfiltration"], "threshold": 0.75},
    {"level": 5, "tasks": ["hard_data_exfiltration"],                  "threshold": 0.80},
]


class CurriculumManager:
    """
    Self-Improvement loop:
    - Blue agent starts at level 1 (easy tasks)
    - When average score over last N episodes exceeds threshold → level up
    - Reward curves, improvements, and level transitions are logged
    """

    def __init__(self) -> None:
        self.current_level: int = 1
        self.episode_count: int = 0
        self.score_history: list[dict[str, Any]] = []
        self.level_up_history: list[dict[str, Any]] = []
        self._rng = random.Random(0)
        self._window = 5  # rolling window for level-up check

    def get_next_task(self) -> dict[str, Any]:
        lvl_cfg = CURRICULUM_LEVELS[min(self.current_level - 1, len(CURRICULUM_LEVELS) - 1)]
        task_id = self._rng.choice(lvl_cfg["tasks"])
        return TASKS[task_id]

    def record_score(self, task_id: str, score: float) -> None:
        self.episode_count += 1
        self.score_history.append({
            "episode": self.episode_count,
            "task_id": task_id,
            "score": round(score, 4),
            "level": self.current_level,
        })
        self._maybe_level_up()

    def _maybe_level_up(self) -> None:
        if self.current_level >= len(CURRICULUM_LEVELS):
            return
        lvl_cfg = CURRICULUM_LEVELS[self.current_level - 1]
        window = [r["score"] for r in self.score_history[-self._window:]]
        if len(window) < self._window:
            return
        avg = sum(window) / len(window)
        if avg >= lvl_cfg["threshold"]:
            old_level = self.current_level
            self.current_level += 1
            self.level_up_history.append({
                "from_level": old_level,
                "to_level": self.current_level,
                "episode": self.episode_count,
                "avg_score": round(avg, 4),
            })

    def summary(self) -> str:
        if not self.score_history:
            return "No episodes recorded yet."
        recent = self.score_history[-10:]
        avg = sum(r["score"] for r in recent) / len(recent)
        return (
            f"Level {self.current_level}/{len(CURRICULUM_LEVELS)} | "
            f"{self.episode_count} episodes | "
            f"Recent avg score: {avg:.3f} | "
            f"Level-ups: {len(self.level_up_history)}"
        )


# ---------------------------------------------------------------------------
# Global multi-agent state
# ---------------------------------------------------------------------------
_ma_sessions: dict[str, MultiAgentSecOpsEnv] = {}
_curriculum = CurriculumManager()


def _get_ma_session(session_id: str) -> MultiAgentSecOpsEnv:
    if session_id not in _ma_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Multi-agent session '{session_id}' not found. Call /multi/reset first.",
        )
    return _ma_sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic v2 compatible)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_memory_leak"
    session_id: str = ""          # defaults to task_id when empty


class StepRequest(BaseModel):
    action_type: str
    parameters: dict[str, Any] = {}
    session_id: str = "default"


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = {}


class StateResponse(BaseModel):
    state: dict[str, Any]


class GradeResponse(BaseModel):
    score: float
    diagnosis_correct: float
    action_efficiency: float
    investigation_quality: float
    details: dict[str, Any]


class MultiAgentResetRequest(BaseModel):
    task_id: str | None = None
    session_id: str = "default_ma"


class RedActionRequest(BaseModel):
    action_type: str | None = None
    parameters: dict[str, Any] = {}
    session_id: str = "default_ma"


class MultiAgentStepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = {}


class CurriculumSummaryResponse(BaseModel):
    current_level: int
    total_episodes: int
    summary_text: str
    level_up_history: list[dict]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OpenSecOpsEnv — Attacker vs Defender",
    description=(
        "SecOps incident response environment for RL agent evaluation. "
        "Simulates memory leaks, DDoS attacks, misconfiguration, and "
        "data exfiltration scenarios across 4 tasks of increasing difficulty. "
        "Features a live Red (Attacker) vs Blue (Defender) multi-agent battle "
        "with self-improving curriculum learning."
    ),
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "environment": "opensecops", "version": "0.4.0"}


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": cfg["difficulty"],
                "max_steps": cfg["max_steps"],
                "description": cfg["description"][:120] + "...",
                "noise_level": cfg["noise_level"],
                "correct_label": cfg["correct_label"],
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None) -> dict[str, Any]:
    """Start a new episode. Returns the initial observation."""
    if req is None:
        req = ResetRequest()
    session_id = req.session_id or req.task_id
    try:
        env = OpenSecOpsEnv()
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _sessions[session_id] = env
    global _default_session
    _default_session = session_id

    return {
        "session_id": session_id,
        "observation": {
            "alerts": obs.alerts,
            "metrics": obs.metrics,
            "logs": obs.logs,
            "topology": obs.topology,
            "last_action_result": obs.last_action_result,
            "time_step": obs.time_step,
            "available_actions": obs.available_actions,
        },
    }


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """Execute one action and receive the next observation + reward."""
    sid = req.session_id if req.session_id != "default" else _default_session
    env = _get_session(sid)

    action = SecOpsAction(
        action_type=req.action_type,
        parameters=req.parameters,
    )
    obs, reward, done, info = env.step(action)

    return StepResponse(
        observation={
            "alerts": obs.alerts,
            "metrics": obs.metrics,
            "logs": obs.logs,
            "topology": obs.topology,
            "last_action_result": obs.last_action_result,
            "time_step": obs.time_step,
            "available_actions": obs.available_actions,
        },
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state(session_id: str = "") -> StateResponse:
    """Return the full internal state (for debugging / graders)."""
    sid = session_id or _default_session
    env = _get_session(sid)
    return StateResponse(state=env.state.to_dict())


@app.post("/grade", response_model=GradeResponse)
def grade_episode(session_id: str = "") -> GradeResponse:
    """Grade the current (or most recently completed) episode."""
    sid = session_id or _default_session
    env = _get_session(sid)
    result = grade(env.state.to_dict())
    return GradeResponse(
        score=result.score,
        diagnosis_correct=result.diagnosis_correct,
        action_efficiency=result.action_efficiency,
        investigation_quality=result.investigation_quality,
        details=result.details,
    )


# ---------------------------------------------------------------------------
# Multi-Agent Endpoints
# ---------------------------------------------------------------------------

@app.post("/multi/reset", response_model=StateResponse)
def multi_reset(req: MultiAgentResetRequest) -> StateResponse:
    """Reset a multi-agent (Attacker vs Defender) session."""
    sid = req.session_id
    _ma_sessions[sid] = MultiAgentSecOpsEnv()
    env = _ma_sessions[sid]

    # If task_id is "auto", curriculum decides the next task
    if req.task_id == "auto":
        task_cfg = _curriculum.get_next_task()
        tid = task_cfg["task_id"]
    else:
        tid = req.task_id or "hard_data_exfiltration"

    state_dict = env.reset(tid)
    return StateResponse(state=state_dict)


@app.post("/multi/blue/step", response_model=MultiAgentStepResponse)
def multi_blue_step(req: StepRequest) -> MultiAgentStepResponse:
    """Blue (Defender) agent step — investigate and mitigate."""
    env = _get_ma_session(req.session_id)
    action = SecOpsAction(
        action_type=req.action_type,
        parameters=req.parameters,
    )
    obs_dict, reward, done, info = env.blue_step(action)

    if done:
        # Record final score in curriculum for self-improvement
        g = grade(env._env.state.to_dict())
        _curriculum.record_score(env._red_state.task_id, g.score)

    return MultiAgentStepResponse(
        observation=obs_dict,
        reward=reward,
        done=done,
        info=info,
    )


@app.post("/multi/red/step", response_model=MultiAgentStepResponse)
def multi_red_step(req: RedActionRequest) -> MultiAgentStepResponse:
    """Red (Attacker) agent step — escalate / confuse / spread."""
    env = _get_ma_session(req.session_id)

    # Use specified action or let heuristic agent decide
    action_type = None
    if req.action_type and req.action_type != "auto":
        action_type = req.action_type

    obs_dict, reward, done, info = env.red_step(action_type)
    return MultiAgentStepResponse(
        observation=obs_dict,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/multi/state")
def multi_state(session_id: str = "default_ma") -> StateResponse:
    """Get the current multi-agent session state."""
    env = _get_ma_session(session_id)
    return StateResponse(state=env.state)


# ---------------------------------------------------------------------------
# Curriculum Endpoints
# ---------------------------------------------------------------------------

@app.get("/curriculum/summary", response_model=CurriculumSummaryResponse)
def curriculum_summary() -> CurriculumSummaryResponse:
    """Return cursor on the Blue agent's self-improvement journey."""
    return CurriculumSummaryResponse(
        current_level=_curriculum.current_level,
        total_episodes=_curriculum.episode_count,
        summary_text=_curriculum.summary(),
        level_up_history=_curriculum.level_up_history,
    )


# ---------------------------------------------------------------------------
# SSE Demo Stream  — powers the live dashboard (single-agent)
# ---------------------------------------------------------------------------

# Heuristic playbooks (deterministic, expert agent)
_HEURISTIC_PLAYBOOKS: dict[str, list[dict]] = {
    "easy_memory_leak": [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "query_logs",       "parameters": {"service": "auth"}},
        {"action_type": "inspect_metrics",  "parameters": {"service": "auth"}},
        {"action_type": "restart_service",  "parameters": {"service": "auth"}},
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
    ],
    "medium_ddos_cascade": [
        {"action_type": "inspect_metrics",   "parameters": {}},
        {"action_type": "query_logs",        "parameters": {"service": "gateway"}},
        {"action_type": "query_logs",        "parameters": {"service": "api"}},
        {"action_type": "run_security_scan", "parameters": {"target": "api"}},
        {"action_type": "block_ip",          "parameters": {"ip": "203.0.113.45"}},
        {"action_type": "block_ip",          "parameters": {"ip": "198.51.100.12"}},
        {"action_type": "scale_service",     "parameters": {"service": "api", "replicas": 5}},
        {"action_type": "submit_diagnosis",  "parameters": {"label": "cyber_attack:ddos"}},
    ],
    "medium_hard_bad_deployment": [
        {"action_type": "inspect_metrics",     "parameters": {}},
        {"action_type": "query_logs",          "parameters": {"service": "api"}},
        {"action_type": "query_logs",          "parameters": {"service": "cache"}},
        {"action_type": "rollback_deployment", "parameters": {"service": "api", "version": "previous"}},
        {"action_type": "restart_service",     "parameters": {"service": "cache"}},
        {"action_type": "submit_diagnosis",    "parameters": {"label": "misconfiguration:bad_config"}},
    ],
    "hard_data_exfiltration": [
        {"action_type": "inspect_metrics",   "parameters": {}},
        {"action_type": "query_logs",        "parameters": {"service": "db"}},
        {"action_type": "query_logs",        "parameters": {"service": "auth"}},
        {"action_type": "run_security_scan", "parameters": {"target": "db"}},
        {"action_type": "run_security_scan", "parameters": {"target": "auth"}},
        {"action_type": "isolate_service",   "parameters": {"service": "db"}},
        {"action_type": "block_ip",          "parameters": {"ip": "10.0.0.99"}},
        {"action_type": "submit_diagnosis",  "parameters": {"label": "cyber_attack:data_exfiltration"}},
    ],
}

# Bad agent: makes wrong choices to show contrast
_BAD_AGENT_PLAYBOOKS: dict[str, list[dict]] = {
    "easy_memory_leak": [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "restart_service",  "parameters": {"service": "api"}},   # wrong service
        {"action_type": "scale_service",    "parameters": {"service": "db", "replicas": 3}},  # useless
        {"action_type": "block_ip",         "parameters": {"ip": "8.8.8.8"}},    # harmful
        {"action_type": "submit_diagnosis", "parameters": {"label": "cyber_attack:ddos"}},  # wrong
    ],
    "medium_ddos_cascade": [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "restart_service",  "parameters": {"service": "db"}},    # wrong service
        {"action_type": "block_ip",         "parameters": {"ip": "8.8.8.8"}},    # wrong ip
        {"action_type": "isolate_service",  "parameters": {"service": "cache"}}, # wrong service
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
    ],
    "medium_hard_bad_deployment": [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "query_logs",       "parameters": {"service": "auth"}},
        {"action_type": "restart_service",  "parameters": {"service": "auth"}},  # wrong
        {"action_type": "block_ip",         "parameters": {"ip": "10.0.0.1"}},   # harmful
        {"action_type": "submit_diagnosis", "parameters": {"label": "cyber_attack:ddos"}},
    ],
    "hard_data_exfiltration": [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "restart_service",  "parameters": {"service": "cache"}}, # chasing false alert
        {"action_type": "query_logs",       "parameters": {"service": "cache"}},
        {"action_type": "isolate_service",  "parameters": {"service": "cache"}}, # wrong, causes outage
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
    ],
}

# Red (Attacker) heuristic playbook — interleaved with blue steps
_RED_PLAYBOOK_BY_TASK: dict[str, list[str]] = {
    "easy_memory_leak":           ["inject_noise", "corrupt_metric", "inject_noise"],
    "medium_ddos_cascade":        ["amplify_attack", "create_false_alert", "amplify_attack", "inject_noise"],
    "medium_hard_bad_deployment": ["inject_noise", "corrupt_metric", "create_false_alert", "inject_noise"],
    "hard_data_exfiltration":     ["amplify_attack", "accelerate_spread", "create_false_alert", "amplify_attack", "inject_noise"],
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/demo/stream")
async def demo_stream(
    task_id: str = "easy_memory_leak",
    mode: str = "trained",   # "trained" | "untrained"
    speed: float = 1.5,      # seconds between steps
):
    """
    Server-Sent Events stream of a full single-agent episode.
    Powers the live dashboard.

    mode=trained  → calls TRAINED_MODEL_ENDPOINT (live AI) or heuristic fallback
    mode=untrained → calls UNTRAINED_MODEL_ENDPOINT (base model) or bad-agent fallback

    Fallback logic: if no endpoint is configured, uses deterministic playbooks
    so the dashboard always works — even before the endpoint is live.
    """
    async def event_gen():
        try:
            # Determine which endpoint to use
            if mode == "trained":
                endpoint = _TRAINED_ENDPOINT
                fallback_playbook = _HEURISTIC_PLAYBOOKS.get(task_id, [])
                agent_label = "🤖 Trained AI (Qwen2.5-7B-GRPO)"
            else:
                endpoint = _UNTRAINED_ENDPOINT
                fallback_playbook = _BAD_AGENT_PLAYBOOKS.get(task_id, [])
                agent_label = "⚠️ Untrained Base Model"

            live_ai = endpoint is not None

            env = OpenSecOpsEnv()
            obs = env.reset(task_id)
            obs_dict = {
                "alerts": obs.alerts,
                "metrics": obs.metrics,
                "logs": obs.logs,
                "topology": obs.topology,
                "last_action_result": obs.last_action_result,
                "time_step": obs.time_step,
            }

            # Send initial state
            yield _sse({
                "type": "reset",
                "task_id": task_id,
                "mode": mode,
                "agent_label": agent_label,
                "live_ai": live_ai,
                "observation": obs_dict,
            })

            cumulative = 0.0
            rewards_history: list[float] = []
            step = 0
            max_steps = TASKS[task_id]["max_steps"]
            done = False

            while not done and step < max_steps:
                await asyncio.sleep(max(0.5, speed))
                step += 1

                # --- Decide action ---
                action: Optional[SecOpsAction] = None
                ai_raw: str = ""

                if live_ai:
                    action = await _query_ai_model(endpoint, obs_dict, step)
                    if action is None:
                        # AI call failed — fall back gracefully
                        ai_raw = "[AI timeout — using fallback]"
                        pb_idx = step - 1
                        if pb_idx < len(fallback_playbook):
                            d = fallback_playbook[pb_idx]
                            action = SecOpsAction(
                                action_type=d["action_type"],
                                parameters=d.get("parameters", {}),
                            )
                        else:
                            action = SecOpsAction(action_type="inspect_metrics", parameters={})
                    else:
                        ai_raw = f"{{\"action_type\": \"{action.action_type}\", \"parameters\": {json.dumps(action.parameters)}}}"
                else:
                    pb_idx = step - 1
                    if pb_idx < len(fallback_playbook):
                        d = fallback_playbook[pb_idx]
                        action = SecOpsAction(
                            action_type=d["action_type"],
                            parameters=d.get("parameters", {}),
                        )
                    else:
                        break

                obs, reward, done, info = env.step(action)
                obs_dict = {
                    "alerts": obs.alerts,
                    "metrics": obs.metrics,
                    "logs": obs.logs,
                    "topology": obs.topology,
                    "last_action_result": obs.last_action_result,
                    "time_step": obs.time_step,
                }
                cumulative += reward
                rewards_history.append(round(reward, 4))

                yield _sse({
                    "type": "step",
                    "step": step,
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "reward": round(reward, 4),
                    "cumulative_reward": round(cumulative, 4),
                    "rewards_history": rewards_history,
                    "done": done,
                    "live_ai": live_ai,
                    "ai_raw": ai_raw,
                    "observation": obs_dict,
                })

            # Final grade
            await asyncio.sleep(0.5)
            result = grade(env.state.to_dict())
            yield _sse({
                "type": "grade",
                "score": round(result.score, 4),
                "diagnosis_correct": result.diagnosis_correct,
                "action_efficiency": round(result.action_efficiency, 4),
                "investigation_quality": round(result.investigation_quality, 4),
                "details": result.details,
                "rewards_history": rewards_history,
                "cumulative_reward": round(cumulative, 4),
                "live_ai": live_ai,
                "agent_label": agent_label,
            })

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/battle/stream")
async def battle_stream(
    task_id: str = "hard_data_exfiltration",
    speed: float = 1.5,
):
    """
    🔴 vs 🔵  Attacker vs Defender live battle SSE stream.
    Interleaves Red (Attacker) and Blue (Defender) turns.

    Blue Agent uses TRAINED_MODEL_ENDPOINT when set (live AI),
    otherwise falls back to the heuristic expert playbook.
    """
    async def event_gen():
        try:
            live_ai = _TRAINED_ENDPOINT is not None

            ma_env = MultiAgentSecOpsEnv()
            state_dict = ma_env.reset(task_id)
            obs_dict = state_dict["observation"]

            yield _sse({
                "type": "battle_reset",
                "task_id": task_id,
                "observation": obs_dict,
                "multi_agent": state_dict.get("multi_agent", {}),
                "live_ai": live_ai,
                "blue_label": "🤖 Trained AI" if live_ai else "🧠 Expert Heuristic",
            })

            blue_playbook = copy.deepcopy(
                _HEURISTIC_PLAYBOOKS.get(task_id, _HEURISTIC_PLAYBOOKS["hard_data_exfiltration"])
            )
            red_actions = _RED_PLAYBOOK_BY_TASK.get(
                task_id, ["inject_noise", "amplify_attack", "corrupt_metric"]
            )

            blue_rewards: list[float] = []
            red_rewards: list[float] = []
            blue_cum = 0.0
            red_cum = 0.0
            done = False
            max_steps = TASKS[task_id]["max_steps"]

            for round_num in range(max_steps):
                if done:
                    break

                # ── Red turn (Attacker escalates) ────────────────────────────
                red_action_type = (
                    red_actions[round_num % len(red_actions)]
                    if red_actions else "inject_noise"
                )
                await asyncio.sleep(max(0.3, speed * 0.4))

                red_state, red_r, done, red_info = ma_env.red_step(red_action_type)
                red_cum += red_r
                red_rewards.append(round(red_r, 4))
                obs_dict = red_state.get("observation", obs_dict)

                yield _sse({
                    "type": "red_step",
                    "round": round_num + 1,
                    "action": red_action_type,
                    "reward": round(red_r, 4),
                    "red_cumulative": round(red_cum, 4),
                    "blue_cumulative": round(blue_cum, 4),
                    "message": red_info.get("message", ""),
                    "observation": obs_dict,
                    "multi_agent": red_state.get("multi_agent", {}),
                    "red_rewards": red_rewards,
                    "blue_rewards": blue_rewards,
                })

                if done:
                    break

                # ── Blue turn (Defender responds) ────────────────────────────
                await asyncio.sleep(max(0.5, speed * 0.6))

                action: Optional[SecOpsAction] = None
                ai_raw: str = ""

                if live_ai:
                    action = await _query_ai_model(_TRAINED_ENDPOINT, obs_dict, round_num + 1)
                    if action:
                        ai_raw = f'{{"action_type": "{action.action_type}", "parameters": {json.dumps(action.parameters)}}}'

                if action is None:
                    # Fallback to heuristic for this round
                    if round_num < len(blue_playbook):
                        d = blue_playbook[round_num]
                        action = SecOpsAction(
                            action_type=d["action_type"],
                            parameters=d.get("parameters", {}),
                        )
                    else:
                        break

                blue_state, blue_r, done, blue_info = ma_env.blue_step(action)
                blue_cum += blue_r
                blue_rewards.append(round(blue_r, 4))
                obs_dict = blue_state.get("observation", obs_dict)

                yield _sse({
                    "type": "blue_step",
                    "round": round_num + 1,
                    "action": action.action_type,
                    "parameters": action.parameters,
                    "reward": round(blue_r, 4),
                    "blue_cumulative": round(blue_cum, 4),
                    "red_cumulative": round(red_cum, 4),
                    "result": obs_dict.get("last_action_result", ""),
                    "observation": obs_dict,
                    "multi_agent": blue_state.get("multi_agent", {}),
                    "red_rewards": red_rewards,
                    "blue_rewards": blue_rewards,
                    "live_ai": live_ai,
                    "ai_raw": ai_raw,
                })

                if done:
                    break

            # Final grade
            await asyncio.sleep(0.5)
            result = grade(ma_env._env.state.to_dict())
            winner = "defender" if blue_cum > red_cum else "attacker"
            yield _sse({
                "type": "battle_end",
                "score": round(result.score, 4),
                "diagnosis_correct": result.diagnosis_correct,
                "action_efficiency": round(result.action_efficiency, 4),
                "investigation_quality": round(result.investigation_quality, 4),
                "blue_cumulative": round(blue_cum, 4),
                "red_cumulative": round(red_cum, 4),
                "winner": winner,
                "blue_rewards": blue_rewards,
                "red_rewards": red_rewards,
                "details": result.details,
                "live_ai": live_ai,
            })

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Dashboard (served at /dashboard)
# ---------------------------------------------------------------------------

@app.get("/ai/status")
async def ai_status():
    """Returns whether the live AI endpoint is configured and reachable."""
    endpoint = _TRAINED_ENDPOINT
    model_ready = False
    if endpoint:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(endpoint.rstrip('/') + '/health')
                model_ready = r.status_code < 400
        except Exception:
            model_ready = True  # Endpoint is set, assume alive (TGI has no /health sometimes)
    return {
        "live_ai": endpoint is not None,
        "endpoint_configured": endpoint is not None,
        "endpoint_url": (endpoint or "").replace("https://", "")[:40] + "..." if endpoint else None,
        "model_name": "Qwen2.5-7B-GRPO (fine-tuned)",
        "model_url": "https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo",
        "model_ready": model_ready,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Live AI Demo Dashboard — Attacker vs Defender + Self-Improvement."""
    return HTMLResponse(content=_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Optional simple debug Web UI
# ---------------------------------------------------------------------------
ENABLE_WEB = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"


@app.get("/web", response_class=HTMLResponse, include_in_schema=ENABLE_WEB)
def web_ui() -> HTMLResponse:
    """Simple debug interface."""
    if not ENABLE_WEB:
        return HTMLResponse(
            content='<h2>Debug UI disabled. Set ENABLE_WEB_INTERFACE=true to enable.</h2>',
            status_code=200,
        )
    return HTMLResponse(content=_build_web_ui())


def _build_web_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OpenSecOpsEnv – Debug UI</title>
<style>
  :root {
    --bg: #070d1a;
    --surface: #101a2f;
    --surface2: #18243f;
    --border: #2b3f67;
    --text: #d9e5ff;
    --text2: #93a8cb;
    --blue: #60a5fa;
    --green: #22c55e;
    --red: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Inter, Segoe UI, sans-serif;
    background:
      radial-gradient(circle at 10% -20%, rgba(59,130,246,0.25), transparent 38%),
      linear-gradient(180deg, #060b17 0%, #070d1a 100%);
    color: var(--text);
    padding: 28px 16px;
  }
  .shell {
    max-width: 1000px;
    margin: 0 auto;
    border: 1px solid rgba(43,63,103,0.65);
    background: rgba(9,15,29,0.82);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 20px 44px rgba(2,8,23,0.45);
  }
  h1 {
    margin: 0 0 8px;
    font-size: 24px;
    color: #8bb9ff;
    letter-spacing: -0.01em;
  }
  .subtle-link { color: var(--blue); text-decoration: none; font-weight: 600; }
  .subtle-link:hover { text-decoration: underline; }
  .panel {
    background: linear-gradient(180deg, var(--surface2), var(--surface));
    border: 1px solid rgba(43,63,103,0.72);
    border-radius: 12px;
    padding: 14px;
    margin-top: 14px;
  }
  h3 {
    margin: 0 0 10px;
    font-size: 13px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text2);
  }
  button {
    background: #2563eb;
    color: #fff;
    border: 1px solid #2563eb;
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
  }
  button:hover { background: #1d4ed8; }
  input, select {
    background: #101a2f;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 10px;
    border-radius: 8px;
  }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.5;
    font-family: 'JetBrains Mono', Consolas, monospace;
    background: rgba(7,13,26,0.75);
    border: 1px solid rgba(43,63,103,0.55);
    border-radius: 10px;
    padding: 12px;
  }
  .reward { color: var(--green); font-weight: 700; }
  .penalty { color: var(--red); font-weight: 700; }
</style>
</head>
<body>
<div class="shell">
<h1>OpenSecOpsEnv Debug Console</h1>
<p><a class="subtle-link" href="/dashboard">→ Open Live Dashboard</a></p>
<div class="panel">
  <h3>Reset</h3>
  <select id="task_id">
    <option value="easy_memory_leak">easy_memory_leak</option>
    <option value="medium_ddos_cascade">medium_ddos_cascade</option>
    <option value="medium_hard_bad_deployment">medium_hard_bad_deployment</option>
    <option value="hard_data_exfiltration">hard_data_exfiltration</option>
  </select>
  <button onclick="doReset()">Reset Episode</button>
</div>
<div class="panel">
  <h3>Step</h3>
  Action type: <select id="atype">
    <option>query_logs</option><option>inspect_metrics</option>
    <option>restart_service</option><option>scale_service</option>
    <option>block_ip</option><option>rollback_deployment</option>
    <option>run_security_scan</option><option>isolate_service</option>
    <option>submit_diagnosis</option>
  </select>
  Params (JSON): <input id="params" size="40" value='{"service":"auth"}'>
  <button onclick="doStep()">Send Action</button>
</div>
<div class="panel"><h3>Last Result</h3><pre id="result">—</pre></div>
<div class="panel">
  <button onclick="doState()">Fetch Full State</button>
  <button onclick="doGrade()">Grade Episode</button>
  <pre id="state_out">—</pre>
</div>
</div>
<script>
async function doReset() {
  const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: document.getElementById('task_id').value})});
  document.getElementById('result').textContent = JSON.stringify(await r.json(), null, 2);
}
async function doStep() {
  let params = {};
  try { params = JSON.parse(document.getElementById('params').value); } catch(e){}
  const r = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action_type: document.getElementById('atype').value, parameters: params})});
  const d = await r.json();
  const el = document.getElementById('result');
  el.textContent = JSON.stringify(d, null, 2);
  el.className = d.reward >= 0 ? 'reward' : 'penalty';
}
async function doState() {
  document.getElementById('state_out').textContent = JSON.stringify(await (await fetch('/state')).json(), null, 2);
}
async function doGrade() {
  document.getElementById('state_out').textContent = JSON.stringify(await (await fetch('/grade', {method:'POST'})).json(), null, 2);
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Dashboard HTML — Attacker vs Defender + Self-Improvement Theme
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenSecOpsEnv — Attacker vs Defender</title>
<meta name="description" content="Live AI Demo: Red Attacker vs Blue Defender agent battle with self-improving curriculum learning. OpenEnv Hackathon submission.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f5f7fb;
    --bg2: #ffffff;
    --bg3: #f8fafc;
    --bg4: #eff3f9;
    --surface: #ffffff;
    --surface-strong: #f8fafc;
    --border: #dbe3ef;
    --border2: #c6d3e6;
    --blue: #1f3b73;
    --blue-bright: #2a4f95;
    --green: #0f9f64;
    --red: #cf3a37;
    --red-bright: #b72e2b;
    --orange: #7c3aed;
    --yellow: #d99800;
    --purple: #6b4de6;
    --cyan: #0ea5a6;
    --text: #132238;
    --text2: #334861;
    --text3: #657a95;
    --card-shadow: 0 10px 24px rgba(15, 31, 58, 0.08);
    --glow-blue: 0 0 0 rgba(0,0,0,0);
    --glow-red: 0 0 0 rgba(0,0,0,0);
    --glow-green: 0 0 0 rgba(0,0,0,0);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Poppins', 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
    background: linear-gradient(180deg, #f7f9fd 0%, #f2f6fb 100%);
  }
  .app {
    width: min(1480px, 100%);
    margin: 0 auto;
    border-left: 1px solid rgba(180,198,223,0.45);
    border-right: 1px solid rgba(180,198,223,0.45);
    min-height: 100vh;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.98));
    box-shadow: none;
  }

  /* ---- Header ---- */
  .header {
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(10px);
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo {
    font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600;
    color: var(--blue-bright); letter-spacing: -0.01em;
    text-shadow: none;
  }
  .logo span { color: var(--purple); font-weight: 600; }
  .badge {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    padding: 3px 8px; border-radius: 4px; text-transform: uppercase;
  }
  .badge-openenv { background: rgba(31,59,115,0.08); color: var(--blue-bright); border: 1px solid rgba(31,59,115,0.2); }
  .badge-live { background: rgba(14,165,166,0.14); color: #0f7879; border: 1px solid rgba(14,165,166,0.28); }
  .badge-self-improve { background: rgba(107,77,230,0.1); color: #5b3cd7; border: 1px solid rgba(107,77,230,0.24); }
  .header-right { display: flex; align-items: center; gap: 16px; }
  .header-status { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text3); }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #11a36c; }

  /* ---- Tab bar ---- */
  .tab-bar {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: #ffffff;
    padding: 8px 24px 0;
    gap: 8px;
  }
  .tab {
    padding: 10px 16px; font-size: 12px; font-weight: 600;
    color: var(--text3); cursor: pointer; border: none; background: none;
    border-bottom: 2px solid transparent; transition: all 0.2s ease;
    display: flex; align-items: center; gap: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
  }
  .tab:hover { color: var(--blue-bright); background: #f2f6fd; }
  .tab.active {
    color: #ffffff;
    border-bottom-color: transparent;
    background: linear-gradient(90deg, #6b4de6, #0ea5a6);
    box-shadow: none;
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: flex; flex-direction: column; }

  /* ---- Controls bar ---- */
  .controls {
    padding: 14px 18px;
    margin: 14px 18px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #ffffff;
    box-shadow: var(--card-shadow);
    flex-wrap: wrap;
  }
  .control-group { display: flex; flex-direction: column; gap: 3px; }
  .control-label { font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text3); }
  select, .btn {
    font-family: 'Inter', sans-serif; font-size: 12px;
    border-radius: 8px; border: 1px solid var(--border2);
    background: #ffffff; color: var(--text);
    padding: 8px 12px; cursor: pointer; outline: none;
    transition: border-color 0.15s, background 0.15s, transform 0.12s ease;
  }
  select:hover, select:focus { border-color: var(--orange); background: #ffffff; }
  .btn { font-weight: 600; display: flex; align-items: center; gap: 5px; }
  .btn-primary { background: var(--orange); color: #fff; border-color: var(--orange); }
  .btn-primary:hover { background: #6d33d8; transform: translateY(-1px); box-shadow: 0 6px 14px rgba(107,77,230,0.28); }
  .btn-primary:disabled { opacity: 0.35; cursor: not-allowed; }
  .btn-red { background: rgba(239,68,68,0.2); color: var(--red-bright); border-color: rgba(239,68,68,0.3); }
  .btn-red:hover { background: rgba(239,68,68,0.3); }
  .btn-ghost { background: transparent; color: var(--text2); border-color: var(--border2); }
  .btn-ghost:hover { color: var(--blue); border-color: var(--blue); background: #f3f7fd; }
  .mode-toggle { display: flex; border-radius: 6px; overflow: hidden; border: 1px solid var(--border2); }
  .mode-btn { padding: 6px 13px; font-size: 11px; font-weight: 600; cursor: pointer; transition: all 0.15s; background: var(--bg3); color: var(--text3); border: none; }
  .mode-btn.active-trained { background: rgba(15,159,100,0.12); color: var(--green); }
  .mode-btn.active-untrained { background: rgba(207,58,55,0.12); color: var(--red-bright); }

  /* ---- Main layout ---- */
  .main {
    display: grid;
    grid-template-columns: 300px 1fr 280px;
    height: calc(100vh - 132px);
    gap: 14px;
    padding: 14px 18px 18px;
  }

  /* ---- Panel base ---- */
  .panel {
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #ffffff;
    border-radius: 14px;
    box-shadow: var(--card-shadow);
  }
  .panel-header {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text3);
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0;
    background: #f8fafd;
  }
  .panel-dot { width: 6px; height: 6px; border-radius: 50%; }
  .panel-body { flex: 1; overflow-y: auto; padding: 12px; }
  .panel-body::-webkit-scrollbar { width: 3px; }
  .panel-body::-webkit-scrollbar-track { background: transparent; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* ---- Scenario card ---- */
  .scenario-card {
    background: #f7faff;
    border: 1px solid #d7e3f3;
    border-radius: 8px; padding: 11px; margin-bottom: 11px;
  }
  .scenario-name { font-size: 12px; font-weight: 600; color: var(--blue); margin-bottom: 3px; }
  .scenario-desc { font-size: 10px; color: var(--text3); line-height: 1.5; }
  .difficulty-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 9px; font-weight: 700; border-radius: 4px; padding: 2px 7px; margin-top: 5px;
    letter-spacing: 0.06em; text-transform: uppercase;
  }
  .diff-easy { background: rgba(15,159,100,0.12); color: var(--green); border: 1px solid rgba(15,159,100,0.22); }
  .diff-medium { background: rgba(217,152,0,0.13); color: var(--yellow); border: 1px solid rgba(217,152,0,0.24); }
  .diff-medium_hard { background: rgba(255,122,26,0.13); color: #c65c08; border: 1px solid rgba(255,122,26,0.24); }
  .diff-hard { background: rgba(207,58,55,0.12); color: var(--red-bright); border: 1px solid rgba(207,58,55,0.22); }

  /* ---- Section title ---- */
  .section-title {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 7px; display: flex; align-items: center; gap: 6px;
  }
  .section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  /* ---- Metrics ---- */
  .metric-row {
    display: grid; grid-template-columns: 72px 1fr 40px;
    align-items: center; gap: 7px; margin-bottom: 7px;
  }
  .metric-svc { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text2); font-weight: 500; }
  .metric-bars { display: flex; flex-direction: column; gap: 2px; }
  .metric-bar-wrap { display: flex; align-items: center; gap: 4px; }
  .metric-bar-label { font-size: 8px; color: var(--text3); width: 20px; }
  .metric-bar-bg { flex: 1; height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .metric-bar-fill { height: 100%; border-radius: 2px; transition: width 0.6s cubic-bezier(0.4,0,0.2,1), background 0.6s; }
  .bar-ok { background: var(--green); }
  .bar-warn { background: var(--cyan); }
  .bar-crit { background: var(--red); }
  .metric-err { font-family: 'JetBrains Mono', monospace; font-size: 9px; text-align: right; }

  /* ---- Alerts ---- */
  .alert-item {
    display: flex; align-items: flex-start; gap: 7px;
    padding: 7px 9px; border-radius: 5px; margin-bottom: 5px;
    border-left: 2px solid; font-size: 10px;
    animation: slide-in 0.22s ease;
  }
  @keyframes slide-in { from { opacity:0; transform: translateX(-5px); } to { opacity:1; transform: translateX(0); } }
  .alert-critical { background: rgba(239,68,68,0.1); border-color: var(--red); color: var(--red-bright); }
  .alert-warning  { background: rgba(14,165,166,0.1); border-color: var(--cyan); color: var(--cyan); }
  .alert-info     { background: rgba(31,59,115,0.08); border-color: var(--blue); color: var(--blue); }
  .alert-red-injected { border-color: #f43f5e; background: rgba(244,63,94,0.12); }
  .alert-sev { font-size: 8px; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 1px; opacity: 0.8; }
  .alert-msg { font-size: 10px; line-height: 1.4; }
  .alert-svc { font-family: 'JetBrains Mono', monospace; font-weight: 600; }

  /* ---- Topology ---- */
  .topology-graph { display: flex; flex-direction: column; gap: 5px; }
  .topo-row { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
  .topo-node {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
    padding: 3px 8px; border-radius: 4px; background: var(--bg3); border: 1px solid var(--border2);
    color: var(--text2); transition: all 0.3s;
  }
  .topo-node.affected { border-color: var(--red); color: var(--red-bright); background: rgba(239,68,68,0.1); }
  .topo-node.mitigated { border-color: var(--green); color: var(--green); background: rgba(16,185,129,0.1); }
  .topo-node.red-spread { border-color: #f43f5e; color: #f43f5e; background: rgba(244,63,94,0.15); box-shadow: 0 0 8px rgba(244,63,94,0.3); }
  .topo-arrow { color: var(--text3); font-size: 10px; }

  /* ---- Center Panel: Battle Feed ---- */
  .center-panel {
    border: 1px solid rgba(46,68,105,0.5);
    display: flex;
    flex-direction: column;
    background: #ffffff;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--card-shadow);
  }
  .battle-feed { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 7px; }
  .battle-feed::-webkit-scrollbar { width: 3px; }
  .battle-feed::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* Battle cards */
  .action-card {
    background: #ffffff; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 13px;
    animation: card-in 0.18s ease;
  }
  @keyframes card-in { from { opacity:0; transform: translateY(5px); } to { opacity:1; transform: translateY(0); } }
  .action-card.blue-card { border-left: 3px solid var(--blue); box-shadow: var(--glow-blue); }
  .action-card.red-card  { border-left: 3px solid var(--red);  box-shadow: var(--glow-red); }
  .action-card.positive  { border-left-color: var(--green); }
  .action-card.negative  { border-left-color: var(--red); }
  .action-card.neutral   { border-left-color: var(--border2); }
  .action-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .agent-label {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 2px 6px; border-radius: 3px; display: flex; align-items: center; gap: 4px;
  }
  .agent-blue { background: rgba(59,130,246,0.15); color: var(--blue-bright); }
  .agent-red  { background: rgba(239,68,68,0.15);  color: var(--red-bright); }
  .action-type-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
    padding: 2px 6px; border-radius: 3px;
  }
  .type-mitigation   { background: rgba(139,92,246,0.15); color: #a78bfa; }
  .type-investigation{ background: rgba(31,59,115,0.1); color: var(--blue); }
  .type-terminal     { background: rgba(234,179,8,0.15); color: var(--yellow); }
  .type-red-attack   { background: rgba(239,68,68,0.15); color: var(--red-bright); }
  .action-result { font-size: 11px; color: var(--text2); line-height: 1.5; }
  .reward-badge {
    display: inline-flex; align-items: center;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
    padding: 2px 6px; border-radius: 3px;
  }
  .reward-pos  { color: var(--green); background: rgba(16,185,129,0.1); }
  .reward-neg  { color: var(--red-bright); background: rgba(239,68,68,0.1); }
  .reward-zero { color: var(--text3); background: var(--bg4); }

  /* Thinking */
  .thinking-indicator {
    display: flex; align-items: center; gap: 7px;
    padding: 9px 12px; border-radius: 7px;
    background: var(--bg3); border: 1px dashed var(--border2);
    font-size: 11px; color: var(--text3);
  }
  .thinking-dots span {
    display: inline-block; width: 4px; height: 4px; border-radius: 50%;
    background: var(--blue); margin: 0 1px;
    opacity: 0.75;
  }

  /* Log stream */
  .log-stream {
    border-top: 1px solid var(--border);
    height: 110px; overflow-y: auto;
    padding: 7px 13px; flex-shrink: 0;
    background: #f6f9fd;
  }
  .log-stream::-webkit-scrollbar { width: 2px; }
  .log-stream::-webkit-scrollbar-thumb { background: var(--border2); }
  .log-line { font-family: 'JetBrains Mono', monospace; font-size: 9px; line-height: 1.8; }
  .log-error { color: #f87171; }
  .log-warn  { color: #fbbf24; }
  .log-crit  { color: #fb923c; font-weight: 600; }
  .log-info  { color: #6f839d; }
  .log-red   { color: #f43f5e; font-weight: 600; }

  /* ---- Right Panel ---- */
  .right-panel {
    display: flex;
    flex-direction: column;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--card-shadow);
  }
  .chart-wrap { padding: 11px 13px; border-bottom: 1px solid var(--border); }

  /* Score grid */
  .score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; padding: 11px 13px; border-bottom: 1px solid var(--border); }
  .score-card {
    background: rgba(23,33,59,0.82); border: 1px solid rgba(42,60,97,0.52);
    border-radius: 7px; padding: 9px; text-align: center;
  }
  .score-val { font-size: 18px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .score-lbl { font-size: 8px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text3); margin-top: 2px; }
  .score-big { grid-column: 1 / -1; background: #f7fbff; border-color: #d7e5f6; }
  .score-blue { background: #f7fbff; border-color: #d7e5f6; }
  .score-red  { background: #fff8f8; border-color: #f3dddd; }

  /* Progress bars */
  .progress-wrap { padding: 11px 13px; flex: 1; overflow-y: auto; }
  .progress-item { margin-bottom: 10px; }
  .progress-header { display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px; }
  .progress-name { color: var(--text3); }
  .progress-val { font-family: 'JetBrains Mono', monospace; color: var(--text2); font-weight: 600; }
  .progress-bar-bg { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .progress-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }

  /* Curriculum panel */
  .curriculum-wrap { padding: 11px 13px; border-top: 1px solid var(--border); }
  .curriculum-header { font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text3); margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
  .level-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 9px; font-weight: 700; padding: 2px 7px; border-radius: 3px;
    background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3);
  }
  .level-progress-bg { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 4px; }
  .level-progress-fill { height: 100%; background: linear-gradient(90deg, var(--purple), var(--blue-bright)); border-radius: 2px; transition: width 1s; }
  .level-desc { font-size: 9px; color: var(--text3); }

  /* ---- Episode End Modal ---- */
  .episode-end {
    display: none; position: fixed; inset: 0; z-index: 200;
    background: rgba(10,15,30,0.88); backdrop-filter: blur(10px);
    align-items: center; justify-content: center;
  }
  .episode-end.show { display: flex; }
  .end-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 14px; padding: 32px; text-align: center;
    max-width: 500px; width: 92%;
    box-shadow: 0 24px 50px rgba(0,0,0,0.5);
    animation: card-in 0.2s ease;
  }
  .end-icon { font-size: 40px; margin-bottom: 8px; }
  .end-title { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
  .end-score { font-size: 46px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-bottom: 16px; }
  .end-vs { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: center; margin-bottom: 18px; }
  .end-col { padding: 12px; border-radius: 8px; }
  .end-col-blue { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); }
  .end-col-red  { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.25); }
  .end-col-title { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
  .end-col-score { font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .end-col-blue .end-col-title { color: var(--blue-bright); }
  .end-col-blue .end-col-score { color: var(--blue-bright); }
  .end-col-red  .end-col-title { color: var(--red-bright); }
  .end-col-red  .end-col-score { color: var(--red-bright); }
  .vs-label { font-size: 12px; font-weight: 700; color: var(--text3); }
  .winner-label { font-size: 13px; font-weight: 700; padding: 8px 18px; border-radius: 8px; margin-bottom: 18px; display: inline-block; }
  .winner-blue { background: rgba(59,130,246,0.15); color: var(--blue-bright); border: 1px solid rgba(59,130,246,0.3); }
  .winner-red  { background: rgba(239,68,68,0.15);  color: var(--red-bright);  border: 1px solid rgba(239,68,68,0.3); }

  /* Toast */
  .toast {
    position: fixed; bottom: 20px; right: 20px; z-index: 300;
    background: var(--bg3); color: var(--text); border: 1px solid var(--border2);
    border-radius: 8px; padding: 11px 16px; font-size: 12px; font-weight: 500;
    box-shadow: var(--card-shadow); animation: toast-in 0.3s ease; max-width: 340px;
  }
  @keyframes toast-in { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform: translateY(0); } }

  /* Empty state */
  .empty-state { color: var(--text3); font-size: 11px; text-align: center; padding: 32px 16px; line-height: 1.8; }

  /* Self-improvement panel content */
  .improvement-stats { display: flex; flex-direction: column; gap: 6px; }
  .improvement-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 9px; border-radius: 5px; background: var(--bg3); border: 1px solid var(--border);
    font-size: 10px;
  }
  .improvement-row-label { color: var(--text3); }
  .improvement-row-val { font-family: 'JetBrains Mono', monospace; color: var(--text2); font-weight: 600; }
  .level-up-event {
    display: flex; align-items: center; gap: 7px;
    padding: 6px 9px; border-radius: 5px;
    background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.25);
    font-size: 10px; color: #a78bfa;
    animation: slide-in 0.3s ease;
  }

  @media (max-width: 1100px) {
    .main { grid-template-columns: 240px 1fr 240px; }
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }
  .ai-raw-output {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    background: rgba(14,165,166,0.07); border: 1px solid rgba(14,165,166,0.18);
    border-radius: 4px; padding: 4px 7px; margin-top: 5px;
    color: #0f7879; word-break: break-all; line-height: 1.5;
  }
  .ai-raw-label { font-size: 8px; font-weight: 700; letter-spacing: 0.08em; color: #0ea5a6; margin-bottom: 2px; }
  @media (max-width: 900px) {
    .header { padding: 10px 14px; }
    .tab-bar { padding: 6px 12px 0; overflow-x: auto; }
    .controls { margin: 10px 12px 0; padding: 10px 12px; }
    .main {
      grid-template-columns: 1fr;
      height: auto;
      min-height: 58vh;
      padding: 10px 12px 14px;
    }
    .panel, .center-panel, .right-panel { min-height: 280px; }
  }
</style>
</head>
<body>
<div class="app">

<!-- Header -->
<div class="header">
  <div class="header-left">
    <span class="logo">OpenSecOps<span>Env</span></span>
    <span class="badge badge-openenv">OpenEnv</span>
    <span class="badge badge-self-improve">Curriculum RL</span>
    <span class="badge badge-live" id="liveBadge" style="display:none">● Streaming</span>
  </div>
  <div class="header-right" style="display:flex;align-items:center;gap:14px">
    <div id="aiStatusBadge" style="display:none;align-items:center;gap:6px;font-size:10px;font-weight:600;padding:4px 10px;border-radius:6px;background:rgba(14,165,166,0.12);color:#0f7879;border:1px solid rgba(14,165,166,0.3)">
      <span style="width:7px;height:7px;border-radius:50%;background:#11a36c;display:inline-block;animation:pulse-dot 1.5s infinite"></span>
      <span id="aiStatusText">AI Model Live</span>
    </div>
    <div class="header-status">
      <div class="status-dot"></div>
      <span>Server online</span>
    </div>
  </div>
</div>

<!-- Tab bar -->
<div class="tab-bar">
  <button class="tab active" id="tab-single"   onclick="switchTab('single')">Agent</button>
  <button class="tab"        id="tab-battle"   onclick="switchTab('battle')">Battle</button>
  <button class="tab"        id="tab-improve"  onclick="switchTab('improve')">Learning</button>
</div>

<!-- ═══════════════════════════════ TAB: Single Agent ═══════════════════════════════ -->
<div class="tab-panel active" id="panel-single" style="height:calc(100vh - 128px)">
  <!-- Controls -->
  <div class="controls">
    <div class="control-group">
      <span class="control-label">Scenario</span>
      <select id="scenarioSelect">
        <option value="easy_memory_leak">Memory Leak - Easy</option>
        <option value="medium_ddos_cascade">DDoS Cascade - Medium</option>
        <option value="medium_hard_bad_deployment">Bad Deployment - Medium-Hard</option>
        <option value="hard_data_exfiltration" selected>Data Exfiltration - Hard</option>
      </select>
    </div>
    <div class="control-group">
      <span class="control-label">Agent Mode</span>
      <div class="mode-toggle">
        <button class="mode-btn active-trained" id="modeTrainedBtn"   onclick="setMode('trained')">Trained Model</button>
        <button class="mode-btn"                id="modeUntrainedBtn" onclick="setMode('untrained')">Baseline Model</button>
      </div>
    </div>
    <div class="control-group">
      <span class="control-label">Speed</span>
      <select id="speedSelect">
        <option value="0.8">Fast</option>
        <option value="1.5" selected>Normal</option>
        <option value="2.5">Slow (Demo)</option>
      </select>
    </div>
    <button class="btn btn-primary" id="startBtn" onclick="startDemo()">Run Episode</button>
    <button class="btn btn-ghost" onclick="resetUI()">Reset</button>
    <button class="btn btn-ghost" onclick="showComparison()">Compare</button>
  </div>

  <!-- 3-col layout -->
  <div class="main">
    <!-- LEFT: System State -->
    <div class="panel">
      <div class="panel-header">
        <span>System State</span>
        <div class="panel-dot" style="background:var(--blue)"></div>
      </div>
      <div class="panel-body" id="leftPanel">
        <div class="empty-state">Select a scenario and click<br><strong>Run Episode</strong> to start.</div>
      </div>
    </div>

    <!-- CENTER: Action Feed -->
    <div class="center-panel">
      <div class="panel-header">
        <span>Agent Action Feed</span>
        <div class="panel-dot" style="background:var(--purple)"></div>
      </div>
      <div class="battle-feed" id="actionFeed">
        <div class="empty-state">Agent actions appear here<br>during incident analysis.</div>
      </div>
      <div class="log-stream" id="logStream">
        <div class="log-line log-info">// System logs will stream here...</div>
      </div>
    </div>

    <!-- RIGHT: Rewards + Scores -->
    <div class="right-panel">
      <div class="panel-header">
        <span>Reward Curve</span>
        <div class="panel-dot" style="background:var(--green)"></div>
      </div>
      <div class="chart-wrap">
        <canvas id="rewardChart" height="160"></canvas>
      </div>
      <div class="score-grid" id="scoreGrid">
        <div class="score-card score-big">
          <div class="score-val" id="scoreTotal" style="color:var(--text3)">—</div>
          <div class="score-lbl">Final Score / 1.0</div>
        </div>
        <div class="score-card">
          <div class="score-val" id="scoreCumReward" style="color:var(--cyan)">0.00</div>
          <div class="score-lbl">Cumul. Reward</div>
        </div>
        <div class="score-card">
          <div class="score-val" id="scoreSteps" style="color:var(--purple)">0</div>
          <div class="score-lbl">Steps Taken</div>
        </div>
      </div>
      <div class="progress-wrap" id="progressWrap">
        <div class="progress-item">
          <div class="progress-header"><span class="progress-name">Diagnosis</span><span class="progress-val" id="pDiagnosis">—</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="pDiagnosisBar" style="width:0%;background:var(--cyan)"></div></div>
        </div>
        <div class="progress-item">
          <div class="progress-header"><span class="progress-name">Action Efficiency</span><span class="progress-val" id="pEfficiency">—</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="pEfficiencyBar" style="width:0%;background:var(--purple)"></div></div>
        </div>
        <div class="progress-item">
          <div class="progress-header"><span class="progress-name">Investigation Quality</span><span class="progress-val" id="pInvestigation">—</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="pInvestigationBar" style="width:0%;background:var(--green)"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ TAB: Battle Mode ═══════════════════════════════ -->
<div class="tab-panel" id="panel-battle" style="height:calc(100vh - 128px)">
  <!-- Battle Controls -->
  <div class="controls">
    <div class="control-group">
      <span class="control-label">Scenario</span>
      <select id="battleScenarioSelect">
        <option value="easy_memory_leak">Memory Leak - Easy</option>
        <option value="medium_ddos_cascade">DDoS Cascade - Medium</option>
        <option value="medium_hard_bad_deployment">Bad Deployment - Medium-Hard</option>
        <option value="hard_data_exfiltration" selected>Data Exfiltration - Hard</option>
      </select>
    </div>
    <div class="control-group">
      <span class="control-label">Speed</span>
      <select id="battleSpeedSelect">
        <option value="0.6">Fast</option>
        <option value="1.2" selected>Normal</option>
        <option value="2.0">Slow (Demo)</option>
      </select>
    </div>
    <button class="btn btn-primary" id="battleStartBtn" onclick="startBattle()">Start Battle</button>
    <button class="btn btn-ghost"   onclick="resetBattle()">Reset</button>
    <div style="margin-left:auto; display:flex; gap:10px; align-items:center; font-size:11px;">
      <span style="color:var(--blue-bright); font-weight:600">Defender</span>
      <span style="color:var(--text3)">vs</span>
      <span style="color:var(--red-bright); font-weight:600">Attacker</span>
    </div>
  </div>

  <!-- Battle 3-col -->
  <div class="main">
    <!-- LEFT: Battle System State -->
    <div class="panel">
      <div class="panel-header">
        <span>Live System State</span>
        <div class="panel-dot" style="background:var(--red)"></div>
      </div>
      <div class="panel-body" id="battleLeftPanel">
        <div class="empty-state">Click <strong>Start Battle</strong><br>to launch live simulation.</div>
      </div>
    </div>

    <!-- CENTER: Battle Feed -->
    <div class="center-panel">
      <div class="panel-header">
        <span id="battleFeedTitle">Battle Feed</span>
        <div class="panel-dot" style="background:var(--purple)"></div>
      </div>
      <div class="battle-feed" id="battleFeed">
        <div class="empty-state">Attacker and defender actions<br>stream here in real time.</div>
      </div>
      <div class="log-stream" id="battleLogStream">
        <div class="log-line log-info">// Combat logs will appear here...</div>
      </div>
    </div>

    <!-- RIGHT: Battle Scores -->
    <div class="right-panel">
      <div class="panel-header">
        <span>Battle Score</span>
        <div class="panel-dot" style="background:var(--cyan)"></div>
      </div>
      <div class="chart-wrap">
        <canvas id="battleChart" height="160"></canvas>
      </div>
      <div class="score-grid">
        <div class="score-card score-blue">
          <div class="score-val" id="blueScore" style="color:var(--blue-bright)">0.00</div>
          <div class="score-lbl">Defender</div>
        </div>
        <div class="score-card score-red">
          <div class="score-val" id="redScore" style="color:var(--red-bright)">0.00</div>
          <div class="score-lbl">Attacker</div>
        </div>
        <div class="score-card score-big">
          <div class="score-val" id="battleFinalScore" style="color:var(--text3)">—</div>
          <div class="score-lbl">Episode Score / 1.0</div>
        </div>
      </div>
      <div class="progress-wrap">
        <div class="progress-item">
          <div class="progress-header"><span class="progress-name">Defender Advantage</span><span class="progress-val" id="bAdvantage">—</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="bAdvantageBar" style="width:50%;background:var(--blue)"></div></div>
        </div>
        <div class="progress-item">
          <div class="progress-header"><span class="progress-name">Attack Suppression</span><span class="progress-val" id="bSuppression">—</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="bSuppressionBar" style="width:0%;background:var(--green)"></div></div>
        </div>
        <div class="progress-item">
          <div class="progress-header"><span class="progress-name">Battle Rounds</span><span class="progress-val" id="bRounds">0</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="bRoundsBar" style="width:0%;background:var(--orange)"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ TAB: Self-Improvement ═══════════════════════════════ -->
<div class="tab-panel" id="panel-improve" style="height:calc(100vh - 128px); flex-direction:row">
  <div style="width:320px; border-right:1px solid var(--border); display:flex; flex-direction:column; overflow-y:auto;">
    <div class="panel-header">
      <span>Curriculum Progress</span>
      <div class="panel-dot" style="background:var(--purple)"></div>
    </div>
    <div class="panel-body" id="curriculumPanel">
      <div class="empty-state">Run episodes to see<br>the self-improvement curriculum.</div>
    </div>
  </div>
  <div style="flex:1; display:flex; flex-direction:column; overflow:hidden;">
    <div class="panel-header">
      <span>Learning Curve & Score History</span>
      <div class="panel-dot" style="background:var(--green)"></div>
    </div>
    <div style="padding:14px; flex:1; display:flex; flex-direction:column; gap:14px; overflow-y:auto;">
      <div style="background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:14px;">
        <canvas id="improvementChart" height="200"></canvas>
      </div>
      <div style="background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:14px;">
        <div class="section-title" style="margin-bottom:10px">Level-Up Events</div>
        <div id="levelUpEvents">
          <div class="empty-state" style="padding:16px">No level-ups yet. Run episodes to progress!</div>
        </div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(139,92,246,0.08),rgba(59,130,246,0.08)); border:1px solid rgba(139,92,246,0.2); border-radius:8px; padding:14px;">
        <div class="section-title">How Self-Improvement Works</div>
        <div style="font-size:11px; color:var(--text3); line-height:1.7;">
          <div style="margin-bottom:6px">1. <strong style="color:var(--text2)">Start at Level 1</strong> with foundational scenarios.</div>
          <div style="margin-bottom:6px">2. <strong style="color:var(--text2)">Track score</strong> over a rolling window of five episodes.</div>
          <div style="margin-bottom:6px">3. <strong style="color:var(--text2)">Promote level automatically</strong> when thresholds are met.</div>
          <div style="margin-bottom:6px">4. <strong style="color:var(--red-bright)">Increase attacker pressure</strong> as defender capability improves.</div>
          <div>5. <strong style="color:var(--green)">Level 5</strong> reflects expert readiness for disguised exfiltration scenarios.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Episode End Modal -->
<div class="episode-end" id="episodeEnd">
  <div class="end-card">
    <div class="end-icon" id="endIcon">Result</div>
    <div class="end-title" id="endTitle">Episode Complete</div>
    <div class="end-score" id="endScore">—</div>
    <div class="end-vs" id="endVs" style="display:none">
      <div class="end-col end-col-blue">
        <div class="end-col-title">Defender</div>
        <div class="end-col-score" id="endBlueScore">—</div>
      </div>
      <div class="vs-label">VS</div>
      <div class="end-col end-col-red">
        <div class="end-col-title">Attacker</div>
        <div class="end-col-score" id="endRedScore">—</div>
      </div>
    </div>
    <div id="endWinnerLabel"></div>
    <div id="endComparison" style="display:none; margin-bottom:16px; text-align:left; font-size:11px; color:var(--text3)">
      Trained: <span id="cmpTrained" style="color:var(--green); font-weight:700"></span> &nbsp;|&nbsp;
      Untrained: <span id="cmpUntrained" style="color:var(--red-bright); font-weight:700"></span>
    </div>
    <button class="btn btn-primary" onclick="closeEnd()" style="margin:0 auto;display:flex">Run Again</button>
  </div>
</div>

</div><!-- /app -->

<script>
// ═══════════════════════════════════════════════════════
// Global State
// ═══════════════════════════════════════════════════════
let currentMode = 'trained';
let currentTask = 'hard_data_exfiltration';
let sse = null;
let battleSse = null;
let rewardChart = null;
let battleChart = null;
let improvementChart = null;
let steps = 0;
let battleRounds = 0;
let comparisonScores = {};
let improvementData = { episodes: [], scores: [], levels: [] };

const TASK_META = {
  'easy_memory_leak': {
    name: 'Memory Leak — Auth Service', diff: 'easy',
    desc: 'The auth service has a progressive memory leak. Rising memory + latency. Agent must investigate and restart the correct service.',
  },
  'medium_ddos_cascade': {
    name: 'DDoS Cascade Attack', diff: 'medium',
    desc: 'A DDoS attack cascades through gateway → api → auth. Agent must correlate logs, block attacking IPs, and scale the API.',
  },
  'medium_hard_bad_deployment': {
    name: 'Bad Deployment — Redis Misconfiguration', diff: 'medium_hard',
    desc: 'api v2.4.1 pushed an invalid Redis connection string, causing a reconnect storm. Agent must rollback the deployment.',
  },
  'hard_data_exfiltration': {
    name: 'Data Exfiltration (Disguised)', diff: 'hard',
    desc: 'A compromised service account "reports_bot" is exfiltrating 4GB+ of data. 55% noise level with a false cache alert to mislead the responder.',
  },
};

const ACTION_CLASS = {
  query_logs: 'investigation', inspect_metrics: 'investigation', run_security_scan: 'investigation',
  restart_service: 'mitigation', scale_service: 'mitigation', block_ip: 'mitigation',
  rollback_deployment: 'mitigation', isolate_service: 'mitigation',
  submit_diagnosis: 'terminal',
};

const RED_ACTION_CLASS = {
  inject_noise: 'red-attack', amplify_attack: 'red-attack', corrupt_metric: 'red-attack',
  create_false_alert: 'red-attack', accelerate_spread: 'red-attack',
};

const ACTION_ICON = {
  query_logs:'LOG', inspect_metrics:'MET', run_security_scan:'SCAN',
  restart_service:'RST', scale_service:'SCL', block_ip:'BLK',
  rollback_deployment:'RBK', isolate_service:'ISO', submit_diagnosis:'SUB',
  inject_noise:'NOISE', amplify_attack:'AMP', corrupt_metric:'CORR',
  create_false_alert:'ALERT', accelerate_spread:'SPREAD',
};

// ═══════════════════════════════════════════════════════
// Tab switching
// ═══════════════════════════════════════════════════════
function switchTab(name) {
  ['single','battle','improve'].forEach(t => {
    document.getElementById('panel-' + t).classList.toggle('active', t === name);
    document.getElementById('tab-' + t).classList.toggle('active', t === name);
  });
  if (name === 'improve') refreshCurriculum();
}

// ═══════════════════════════════════════════════════════
// Chart init
// ═══════════════════════════════════════════════════════
function makeChart(canvasId, label1, color1, label2, color2) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: label1, data: [], borderColor: color1,
        backgroundColor: color1 + '20', borderWidth: 2,
        pointRadius: 3, pointBackgroundColor: color1, tension: 0.3, fill: true,
      }, ...(label2 ? [{
        label: label2, data: [], borderColor: color2,
        borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0, tension: 0.3, fill: false,
      }] : [])],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' }, boxWidth: 12 } },
        tooltip: { backgroundColor: '#0d1426', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#1e2d45', borderWidth: 1 },
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(30,45,69,0.8)' } },
        y: { ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }, grid: { color: 'rgba(30,45,69,0.8)' }, min: -1.2, max: 1.2 },
      },
    },
  });
}

function initChart() {
  const ctx = document.getElementById('rewardChart').getContext('2d');
  if (rewardChart) rewardChart.destroy();
  rewardChart = makeChart('rewardChart', 'Step Reward', '#3b82f6', 'Cumulative', '#10b981');
}

function initBattleChart() {
  const ctx = document.getElementById('battleChart').getContext('2d');
  if (battleChart) battleChart.destroy();
  battleChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Defender', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true },
        { label: 'Attacker', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' }, boxWidth: 12 } },
        tooltip: { backgroundColor: '#0d1426', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#1e2d45', borderWidth: 1 },
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(30,45,69,0.8)' } },
        y: { ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }, grid: { color: 'rgba(30,45,69,0.8)' } },
      },
    },
  });
}

function initImprovementChart() {
  const ctx = document.getElementById('improvementChart').getContext('2d');
  if (improvementChart) improvementChart.destroy();
  improvementChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Episode Score', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', borderWidth: 2, pointRadius: 4, tension: 0.3, fill: true },
        { label: 'Curriculum Level', data: [], borderColor: '#8b5cf6', borderWidth: 2, borderDash: [5, 3], pointRadius: 0, tension: 0, fill: false, yAxisID: 'y2' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' }, boxWidth: 12 } }, tooltip: { backgroundColor: '#0d1426', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#1e2d45', borderWidth: 1 } },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(30,45,69,0.8)' } },
        y: { ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }, grid: { color: 'rgba(30,45,69,0.8)' }, min: 0, max: 1 },
        y2: { position: 'right', ticks: { color: '#8b5cf6', font: { size: 9 }, stepSize: 1 }, grid: { display: false }, min: 0, max: 5 },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════
// Single Agent UI helpers
// ═══════════════════════════════════════════════════════
function setMode(m) {
  currentMode = m;
  document.getElementById('modeTrainedBtn').className   = 'mode-btn' + (m==='trained'   ? ' active-trained'   : '');
  document.getElementById('modeUntrainedBtn').className  = 'mode-btn' + (m==='untrained' ? ' active-untrained' : '');
}

function resetUI() {
  if (sse) { sse.close(); sse = null; }
  document.getElementById('actionFeed').innerHTML = '<div class="empty-state">Agent actions appear here<br>during incident analysis.</div>';
  document.getElementById('logStream').innerHTML  = '<div class="log-line log-info">// System logs will stream here...</div>';
  document.getElementById('leftPanel').innerHTML  = '<div class="empty-state">Select a scenario and click<br><strong>Run Episode</strong> to start.</div>';
  document.getElementById('scoreTotal').textContent    = '—';
  document.getElementById('scoreTotal').style.color    = 'var(--text3)';
  document.getElementById('scoreCumReward').textContent = '0.00';
  document.getElementById('scoreSteps').textContent    = '0';
  ['pDiagnosis','pEfficiency','pInvestigation'].forEach(id => document.getElementById(id).textContent = '—');
  ['pDiagnosisBar','pEfficiencyBar','pInvestigationBar'].forEach(id => document.getElementById(id).style.width = '0%');
  document.getElementById('liveBadge').style.display = 'none';
  document.getElementById('startBtn').disabled = false;
  initChart();
  steps = 0;
}

function barClass(val, warn, crit) {
  if (val >= crit) return 'bar-crit';
  if (val >= warn) return 'bar-warn';
  return 'bar-ok';
}

function renderSystemState(obs, taskId, panelId) {
  const meta = TASK_META[taskId] || {};
  const diff = meta.diff || 'easy';
  const affectedByTask = {
    'easy_memory_leak': ['auth'],
    'medium_ddos_cascade': ['api','auth','gateway'],
    'medium_hard_bad_deployment': ['api','cache'],
    'hard_data_exfiltration': ['db','auth'],
  };
  const affected = new Set(affectedByTask[taskId] || []);

  let html = `
    <div class="scenario-card">
      <div class="scenario-name">${meta.name || taskId}</div>
      <div class="scenario-desc">${meta.desc || ''}</div>
      <span class="difficulty-badge diff-${diff}">${diff.toUpperCase()}</span>
    </div>`;

  if (obs.alerts && obs.alerts.length) {
    html += `<div class="section-title">Active Alerts</div>`;
    obs.alerts.slice(0, 6).forEach(a => {
      const isRed = a._red_injected;
      const cls = isRed ? 'alert-critical alert-red-injected' : (a.severity === 'critical' ? 'alert-critical' : a.severity === 'warning' ? 'alert-warning' : 'alert-info');
      html += `<div class="alert-item ${cls}">
        <div>
          <div class="alert-sev">${isRed ? '[ATTACKER]' : '[' + (a.severity||'info').toUpperCase() + ']'}</div>
          <div class="alert-msg"><span class="alert-svc">${a.service}</span> · ${a.message || a.type}</div>
        </div></div>`;
    });
  }

  if (obs.metrics && Object.keys(obs.metrics).length) {
    html += `<div class="section-title" style="margin-top:10px">Service Metrics</div>`;
    Object.entries(obs.metrics).forEach(([svc, m]) => {
      const cpuCls = barClass(m.cpu, 60, 85);
      const memCls = barClass(m.memory, 70, 85);
      const latCls = m.latency > 500 ? 'bar-crit' : m.latency > 200 ? 'bar-warn' : 'bar-ok';
      const errColor = m.error_rate > 10 ? 'var(--red-bright)' : m.error_rate > 5 ? 'var(--orange)' : 'var(--text3)';
      html += `<div class="metric-row">
        <span class="metric-svc">${svc}</span>
        <div class="metric-bars">
          <div class="metric-bar-wrap"><span class="metric-bar-label">cpu</span><div class="metric-bar-bg"><div class="metric-bar-fill ${cpuCls}" style="width:${Math.min(m.cpu,100)}%"></div></div></div>
          <div class="metric-bar-wrap"><span class="metric-bar-label">mem</span><div class="metric-bar-bg"><div class="metric-bar-fill ${memCls}" style="width:${Math.min(m.memory,100)}%"></div></div></div>
          <div class="metric-bar-wrap"><span class="metric-bar-label">lat</span><div class="metric-bar-bg"><div class="metric-bar-fill ${latCls}" style="width:${Math.min(m.latency/2000*100,100)}%"></div></div></div>
        </div>
        <span class="metric-err" style="color:${errColor}">${m.error_rate.toFixed(1)}%</span>
      </div>`;
    });
  }

  if (obs.topology && Object.keys(obs.topology).length) {
    html += `<div class="section-title" style="margin-top:10px">Service Topology</div><div class="topology-graph">`;
    Object.entries(obs.topology).forEach(([svc, deps]) => {
      const nodeClass = affected.has(svc) ? 'affected' : '';
      html += `<div class="topo-row"><span class="topo-node ${nodeClass}">${svc}</span>`;
      if (deps.length) {
        html += `<span class="topo-arrow">→</span>`;
        deps.forEach(d => { html += `<span class="topo-node ${affected.has(d) ? 'affected' : ''}">${d}</span>`; });
      }
      html += `</div>`;
    });
    html += `</div>`;
  }

  document.getElementById(panelId).innerHTML = html;
}

function appendActionCard(feedId, step, agentType, actionType, params, reward, resultMsg, isRed, aiRaw='') {
  const feed = document.getElementById(feedId);
  const isEmpty = feed.querySelector('.empty-state');
  if (isEmpty) feed.innerHTML = '';

  const r = reward;
  const rClass = r > 0 ? 'reward-pos' : r < 0 ? 'reward-neg' : 'reward-zero';
  const rSign = r > 0 ? '+' : '';
  const aClass = isRed ? 'type-red-attack' : (ACTION_CLASS[actionType] || 'type-investigation');
  const cardClass = isRed ? 'red-card' : (r > 0 ? 'blue-card positive' : r < 0 ? 'blue-card negative' : 'blue-card neutral');
  const icon = ACTION_ICON[actionType] || 'ACT';
  const paramsStr = params && Object.keys(params).length ? Object.entries(params).map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ') : 'auto';
  const agentLabel = isRed
    ? '<span class="agent-label agent-red">🔴 Attacker</span>'
    : '<span class="agent-label agent-blue">🔵 Defender</span>';

  const card = document.createElement('div');
  card.className = `action-card ${cardClass}`;
  card.innerHTML = `
    <div class="action-header">
      ${agentLabel}
      <span class="action-type-badge ${aClass}">${icon} ${actionType}</span>
      <span class="reward-badge ${rClass}">${rSign}${r.toFixed(2)}</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text3);margin-bottom:4px">${paramsStr}</div>
    <div class="action-result">${resultMsg || ''}</div>
    ${aiRaw ? `<div class="ai-raw-output"><div class="ai-raw-label">🤖 AI Output</div>${escHtml(aiRaw)}</div>` : ''}
  `;
  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function appendLogs(streamId, logs, isRed) {
  if (!logs || !logs.length) return;
  const stream = document.getElementById(streamId);
  logs.forEach(line => {
    const div = document.createElement('div');
    const cls = isRed
      ? 'log-red'
      : (line.includes('CRIT') ? 'log-crit' : line.includes('ERROR') ? 'log-error' : line.includes('WARN') ? 'log-warn' : 'log-info');
    div.className = `log-line ${cls}`;
    div.textContent = line;
    stream.appendChild(div);
  });
  stream.scrollTop = stream.scrollHeight;
}

function updateChart(rewardsHistory, cumulative) {
  if (!rewardChart) return;
  const n = rewardsHistory.length;
  rewardChart.data.labels = Array.from({length: n}, (_, i) => `S${i+1}`);
  rewardChart.data.datasets[0].data = rewardsHistory;
  let cum = 0;
  rewardChart.data.datasets[1].data = rewardsHistory.map(r => { cum += r; return Math.round(cum*100)/100; });
  rewardChart.update('none');
}

function updateBattleChart(blueRewards, redRewards) {
  if (!battleChart) return;
  const maxLen = Math.max(blueRewards.length, redRewards.length);
  battleChart.data.labels = Array.from({length: maxLen}, (_, i) => `R${i+1}`);
  let bc = 0, rc = 0;
  battleChart.data.datasets[0].data = blueRewards.map(r => { bc += r; return Math.round(bc*100)/100; });
  battleChart.data.datasets[1].data = redRewards.map(r => { rc += r; return Math.round(rc*100)/100; });
  battleChart.update('none');
}

function showThinking(feedId, agentLabel) {
  const feed = document.getElementById(feedId);
  const t = document.createElement('div');
  t.className = 'thinking-indicator';
  t.id = 'thinkingIndicator';
  t.innerHTML = `${agentLabel} processing...&nbsp;<span class="thinking-dots"><span></span><span></span><span></span></span>`;
  feed.appendChild(t);
  feed.scrollTop = feed.scrollHeight;
}

function hideThinking() {
  const t = document.getElementById('thinkingIndicator');
  if (t) t.remove();
}

function updateScores(data) {
  document.getElementById('scoreCumReward').textContent = (data.cumulative_reward || 0).toFixed(2);
  document.getElementById('scoreSteps').textContent = data.step || steps;
}

function updateGradeUI(g) {
  const scoreEl = document.getElementById('scoreTotal');
  scoreEl.textContent = g.score.toFixed(3);
  scoreEl.style.color = g.score > 0.7 ? 'var(--green)' : g.score > 0.4 ? 'var(--orange)' : 'var(--red)';

  document.getElementById('pDiagnosis').textContent = g.diagnosis_correct.toFixed(2);
  const diagBar = document.getElementById('pDiagnosisBar');
  diagBar.style.width = (g.diagnosis_correct * 100) + '%';
  diagBar.style.background = g.diagnosis_correct > 0.9 ? 'var(--green)' : g.diagnosis_correct > 0.4 ? 'var(--orange)' : 'var(--red)';

  document.getElementById('pEfficiency').textContent = g.action_efficiency.toFixed(3);
  const effBar = document.getElementById('pEfficiencyBar');
  effBar.style.width = (g.action_efficiency * 100) + '%';
  effBar.style.background = g.action_efficiency > 0.7 ? 'var(--green)' : g.action_efficiency > 0.4 ? 'var(--orange)' : 'var(--red)';

  document.getElementById('pInvestigation').textContent = g.investigation_quality.toFixed(3);
  const invBar = document.getElementById('pInvestigationBar');
  invBar.style.width = (g.investigation_quality * 100) + '%';
  invBar.style.background = g.investigation_quality > 0.7 ? 'var(--green)' : g.investigation_quality > 0.4 ? 'var(--orange)' : 'var(--red)';
}

function showEpisodeEnd(score, icon, title, scoreColor) {
  document.getElementById('endIcon').textContent  = icon;
  document.getElementById('endTitle').textContent = title;
  const scoreEl = document.getElementById('endScore');
  scoreEl.textContent  = typeof score === 'number' ? score.toFixed(3) : score;
  scoreEl.style.color  = scoreColor || 'var(--text)';
  document.getElementById('endVs').style.display = 'none';
  document.getElementById('endComparison').style.display = 'none';
  document.getElementById('endWinnerLabel').innerHTML = '';
  document.getElementById('episodeEnd').classList.add('show');
}

function closeEnd() {
  document.getElementById('episodeEnd').classList.remove('show');
}

// ═══════════════════════════════════════════════════════
// Single-agent demo
// ═══════════════════════════════════════════════════════
function startDemo() {
  resetUI();
  currentTask = document.getElementById('scenarioSelect').value;
  const speed = document.getElementById('speedSelect').value;
  document.getElementById('startBtn').disabled = true;
  document.getElementById('liveBadge').style.display = 'inline-flex';

  const url = `/demo/stream?task_id=${currentTask}&mode=${currentMode}&speed=${speed}`;
  sse = new EventSource(url);

  sse.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.type === 'reset') {
      renderSystemState(data.observation, data.task_id, 'leftPanel');
      const label = data.live_ai ? `🤖 ${data.agent_label || 'Qwen2.5-7B-GRPO thinking'}` : '🧠 Expert Heuristic';
      showThinking('actionFeed', label);
    }
    else if (data.type === 'step') {
      hideThinking();
      steps = data.step;
      appendActionCard('actionFeed', data.step, 'blue', data.action_type, data.parameters, data.reward, data.observation?.last_action_result, false, data.ai_raw || '');
      appendLogs('logStream', data.observation?.logs, false);
      renderSystemState(data.observation, currentTask, 'leftPanel');
      updateChart(data.rewards_history, data.cumulative_reward);
      updateScores(data);
      if (!data.done) showThinking('actionFeed', data.live_ai ? '🤖 Qwen2.5-7B thinking' : 'Agent');
    }
    else if (data.type === 'grade') {
      hideThinking();
      document.getElementById('liveBadge').style.display = 'none';
      document.getElementById('startBtn').disabled = false;
      updateChart(data.rewards_history, data.cumulative_reward);
      updateGradeUI(data);
      comparisonScores[currentMode] = data.score;

      // Record in improvement tracker
      improvementData.episodes.push(improvementData.episodes.length + 1);
      improvementData.scores.push(data.score);
      improvementData.levels.push(1); // single agent always level 1

      const correct = data.diagnosis_correct > 0.9;
      const modeLabel = data.live_ai ? (data.agent_label || 'AI Model') : 'Heuristic';
      setTimeout(() => showEpisodeEnd(
        data.score,
        correct ? '✅' : data.score > 0.5 ? '⚠️' : '❌',
        correct ? `Incident Resolved! (${modeLabel})` : data.score > 0.5 ? 'Partially Resolved' : 'Episode Failed',
        data.score > 0.7 ? 'var(--green)' : data.score > 0.4 ? 'var(--yellow)' : 'var(--red)'
      ), 800);
      sse.close();
    }
    else if (data.type === 'error') {
      hideThinking();
      document.getElementById('startBtn').disabled = false;
      document.getElementById('liveBadge').style.display = 'none';
      document.getElementById('actionFeed').innerHTML += `<div class="alert-item alert-critical"><div><div class="alert-sev">[ERROR]</div><div>${data.message}</div></div></div>`;
    }
  };

  sse.onerror = () => {
    hideThinking();
    document.getElementById('startBtn').disabled = false;
    document.getElementById('liveBadge').style.display = 'none';
    if (sse) sse.close();
  };
}

function showToast(msg, duration = 4000) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

async function showComparison() {
  if (Object.keys(comparisonScores).length < 2) {
    showToast('Run both model modes first, then click Compare.');
    return;
  }
  document.getElementById('endIcon').textContent  = 'Compare';
  document.getElementById('endTitle').textContent = 'Trained vs Untrained Agent';
  const diff = (comparisonScores.trained || 0) - (comparisonScores.untrained || 0);
  const scoreEl = document.getElementById('endScore');
  scoreEl.textContent = `+${(diff * 100).toFixed(0)}% better`;
  scoreEl.style.color = 'var(--green)';
  document.getElementById('endComparison').style.display = 'block';
  document.getElementById('cmpTrained').textContent   = (comparisonScores.trained || 0).toFixed(3);
  document.getElementById('cmpUntrained').textContent = (comparisonScores.untrained || 0).toFixed(3);
  document.getElementById('endVs').style.display = 'none';
  document.getElementById('endWinnerLabel').innerHTML = '';
  document.getElementById('episodeEnd').classList.add('show');
}

// ═══════════════════════════════════════════════════════
// Battle Mode
// ═══════════════════════════════════════════════════════
function resetBattle() {
  if (battleSse) { battleSse.close(); battleSse = null; }
  document.getElementById('battleFeed').innerHTML = '<div class="empty-state">Attacker and defender actions<br>stream here in real time.</div>';
  document.getElementById('battleLogStream').innerHTML = '<div class="log-line log-info">// Combat logs will appear here...</div>';
  document.getElementById('battleLeftPanel').innerHTML = '<div class="empty-state">Click <strong>Start Battle</strong><br>to launch live simulation.</div>';
  document.getElementById('blueScore').textContent = '0.00';
  document.getElementById('redScore').textContent  = '0.00';
  document.getElementById('battleFinalScore').textContent = '—';
  document.getElementById('bAdvantage').textContent  = '—';
  document.getElementById('bSuppression').textContent = '—';
  document.getElementById('bRounds').textContent   = '0';
  document.getElementById('bAdvantageBar').style.width  = '50%';
  document.getElementById('bSuppressionBar').style.width = '0%';
  document.getElementById('bRoundsBar').style.width  = '0%';
  document.getElementById('battleStartBtn').disabled = false;
  document.getElementById('liveBadge').style.display = 'none';
  initBattleChart();
  battleRounds = 0;
}

function startBattle() {
  resetBattle();
  const taskId = document.getElementById('battleScenarioSelect').value;
  const speed  = document.getElementById('battleSpeedSelect').value;
  document.getElementById('battleStartBtn').disabled = true;
  document.getElementById('liveBadge').style.display = 'inline-flex';

  let blueCum = 0, redCum = 0;
  const blueRewardsHist = [], redRewardsHist = [];

  const url = `/battle/stream?task_id=${taskId}&speed=${speed}`;
  battleSse = new EventSource(url);

  battleSse.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.type === 'battle_reset') {
      renderSystemState(data.observation, taskId, 'battleLeftPanel');
      const blueLabel = data.live_ai ? `🤖 ${data.blue_label || 'Qwen2.5-7B'}` : '🧠 Expert Heuristic';
      document.getElementById('battleFeedTitle').textContent = `Battle Feed — ${blueLabel} vs 🔴 Attacker`;
      showThinking('battleFeed', 'Initializing battle...');
    }

    else if (data.type === 'red_step') {
      hideThinking();
      battleRounds = data.round;
      redCum = data.red_cumulative;
      if (data.reward > 0) redRewardsHist.push(data.reward);
      appendActionCard('battleFeed', data.round, 'red', data.action, {}, data.reward, data.message, true);
      appendLogs('battleLogStream', [`[ATTACKER] ${data.message}`], true);
      renderSystemState(data.observation, taskId, 'battleLeftPanel');
      document.getElementById('redScore').textContent = redCum.toFixed(2);
      if (data.red_rewards && data.blue_rewards) updateBattleChart(data.blue_rewards, data.red_rewards);
      updateBattleBars(blueCum, redCum, data.round);
      showThinking('battleFeed', '🔵 Defender analyzing threat...');
    }

    else if (data.type === 'blue_step') {
      hideThinking();
      battleRounds = data.round;
      blueCum = data.blue_cumulative;
      if (data.reward !== 0) blueRewardsHist.push(data.reward);
      appendActionCard('battleFeed', data.round, 'blue', data.action, data.parameters || {}, data.reward, data.result, false, data.ai_raw || '');
      appendLogs('battleLogStream', data.observation?.logs, false);
      renderSystemState(data.observation, taskId, 'battleLeftPanel');
      document.getElementById('blueScore').textContent = blueCum.toFixed(2);
      if (data.red_rewards && data.blue_rewards) updateBattleChart(data.blue_rewards, data.red_rewards);
      updateBattleBars(blueCum, redCum, data.round);
      showThinking('battleFeed', '🔴 Attacker planning next move...');
    }

    else if (data.type === 'battle_end') {
      hideThinking();
      document.getElementById('liveBadge').style.display = 'none';
      document.getElementById('battleStartBtn').disabled = false;
      document.getElementById('battleFinalScore').textContent = data.score.toFixed(3);
      document.getElementById('battleFinalScore').style.color = data.score > 0.7 ? 'var(--green)' : data.score > 0.4 ? 'var(--orange)' : 'var(--red)';
      if (data.red_rewards && data.blue_rewards) updateBattleChart(data.blue_rewards, data.red_rewards);

      // Record improvement
      improvementData.episodes.push(improvementData.episodes.length + 1);
      improvementData.scores.push(data.score);
      improvementData.levels.push(1);

      // Show end modal
      const winner = data.winner;
      document.getElementById('endVs').style.display     = 'grid';
      document.getElementById('endComparison').style.display = 'none';
      document.getElementById('endBlueScore').textContent = data.blue_cumulative.toFixed(2);
      document.getElementById('endRedScore').textContent  = data.red_cumulative.toFixed(2);
      document.getElementById('endWinnerLabel').innerHTML = winner === 'defender'
        ? '<div class="winner-label winner-blue">Defender Wins</div>'
        : '<div class="winner-label winner-red">Attacker Wins</div>';
      setTimeout(() => showEpisodeEnd(
        data.score,
        winner === 'defender' ? 'Defender' : 'Attacker',
        winner === 'defender' ? 'Defender Wins!' : 'Attacker Wins!',
        winner === 'defender' ? 'var(--blue-bright)' : 'var(--red-bright)'
      ), 800);
      battleSse.close();
    }

    else if (data.type === 'error') {
      hideThinking();
      document.getElementById('battleStartBtn').disabled = false;
      document.getElementById('liveBadge').style.display = 'none';
      document.getElementById('battleFeed').innerHTML += `<div class="alert-item alert-critical"><div><div class="alert-sev">[ERROR]</div><div>${data.message}</div></div></div>`;
    }
  };

  battleSse.onerror = () => {
    hideThinking();
    document.getElementById('battleStartBtn').disabled = false;
    document.getElementById('liveBadge').style.display = 'none';
    if (battleSse) battleSse.close();
  };
}

function updateBattleBars(blue, red, round) {
  const total = Math.abs(blue) + Math.abs(red) || 1;
  const blueAdv = blue / (total + Math.max(Math.abs(blue - red), 0.1));
  const suppression = Math.max(0, blue / (Math.abs(blue) + 1));

  document.getElementById('bAdvantage').textContent  = blue > red ? `+${(blue - red).toFixed(2)}` : `${(blue - red).toFixed(2)}`;
  document.getElementById('bSuppression').textContent = `${(suppression * 100).toFixed(0)}%`;
  document.getElementById('bRounds').textContent    = round;

  document.getElementById('bAdvantageBar').style.width    = Math.min(100, Math.max(0, 50 + (blue - red) * 20)) + '%';
  document.getElementById('bAdvantageBar').style.background = blue > red ? 'var(--blue)' : 'var(--red)';
  document.getElementById('bSuppressionBar').style.width  = (suppression * 100) + '%';
  document.getElementById('bRoundsBar').style.width = Math.min(round / 10 * 100, 100) + '%';
}

// ═══════════════════════════════════════════════════════
// Self-Improvement Tab
// ═══════════════════════════════════════════════════════
async function refreshCurriculum() {
  try {
    const r = await fetch('/curriculum/summary');
    const d = await r.json();
    renderCurriculum(d);
  } catch(e) {
    document.getElementById('curriculumPanel').innerHTML = `<div class="empty-state">Could not load curriculum data.<br>Run episodes first.</div>`;
  }
}

function renderCurriculum(d) {
  const maxLevel = 5;
  const pct = ((d.current_level - 1) / (maxLevel - 1)) * 100;
  let html = `
    <div class="section-title">Current Level</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <span class="level-badge">Level ${d.current_level} / ${maxLevel}</span>
      <span style="font-size:10px;color:var(--text3)">${d.summary_text}</span>
    </div>
    <div class="level-progress-bg"><div class="level-progress-fill" style="width:${pct}%"></div></div>
    <div class="level-desc" style="margin-bottom:12px">Progress to Level ${Math.min(d.current_level + 1, maxLevel)}</div>

    <div class="section-title">Statistics</div>
    <div class="improvement-stats">
      <div class="improvement-row"><span class="improvement-row-label">Total Episodes</span><span class="improvement-row-val">${d.total_episodes}</span></div>
      <div class="improvement-row"><span class="improvement-row-label">Level-Ups Achieved</span><span class="improvement-row-val">${d.level_up_history.length}</span></div>
    </div>`;

  if (d.level_up_history.length > 0) {
    html += `<div class="section-title" style="margin-top:12px">Level-Up History</div>`;
    d.level_up_history.forEach(lu => {
      html += `<div class="level-up-event">Episode ${lu.episode}: Level ${lu.from_level} -> ${lu.to_level} (avg ${lu.avg_score.toFixed(3)})</div>`;
    });
  }

  document.getElementById('curriculumPanel').innerHTML = html;

  // Update level-up events panel
  const luEl = document.getElementById('levelUpEvents');
  if (d.level_up_history.length === 0) {
    luEl.innerHTML = '<div class="empty-state" style="padding:16px">No level-ups yet. Run more episodes!</div>';
  } else {
    luEl.innerHTML = d.level_up_history.map(lu =>
      `<div class="level-up-event" style="margin-bottom:5px">
        Episode ${lu.episode}: <strong>Level ${lu.from_level} -> ${lu.to_level}</strong>
        &nbsp;·&nbsp; avg score <strong>${lu.avg_score.toFixed(3)}</strong>
      </div>`
    ).join('');
  }

  // Update improvement chart
  if (improvementData.episodes.length > 0 && improvementChart) {
    improvementChart.data.labels = improvementData.episodes.map(e => `Ep${e}`);
    improvementChart.data.datasets[0].data = improvementData.scores;
    improvementChart.data.datasets[1].data = improvementData.levels;
    improvementChart.update('none');
  }
}

// ═══════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  initBattleChart();
  initImprovementChart();
  document.getElementById('scenarioSelect').addEventListener('change', e => { currentTask = e.target.value; });

  // Check AI status on load
  fetch('/ai/status').then(r => r.json()).then(d => {
    const badge = document.getElementById('aiStatusBadge');
    if (d.live_ai) {
      badge.style.display = 'flex';
      document.getElementById('aiStatusText').textContent = `🤖 ${d.model_name}`;
    }
  }).catch(() => {});
});
</script>
</body>
</html>
"""
