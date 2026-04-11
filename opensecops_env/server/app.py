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
GET  /web            – interactive web UI (if ENABLE_WEB_INTERFACE=true)

Session model
-------------
Each reset() call creates a named session keyed by `session_id`
(defaults to the task_id). This allows multiple concurrent agents
to run different tasks without colliding.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from opensecops_env.env import OpenSecOpsEnv
from opensecops_env.models import SecOpsAction
from opensecops_env.grader import grade
from opensecops_env.tasks.task_definitions import TASKS

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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OpenSecOpsEnv",
    description=(
        "SecOps incident response environment for RL agent evaluation. "
        "Simulates memory leaks, DDoS attacks, misconfiguration, and "
        "data exfiltration scenarios across 4 tasks of increasing difficulty."
    ),
    version="0.2.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "environment": "opensecops", "version": "0.2.0"}


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
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    """Start a new episode. Returns the initial observation."""
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
# Optional Web UI
# ---------------------------------------------------------------------------
ENABLE_WEB = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"


@app.get("/web", response_class=HTMLResponse, include_in_schema=ENABLE_WEB)
def web_ui() -> HTMLResponse:
    """Interactive web interface for manual environment exploration."""
    if not ENABLE_WEB:
        return HTMLResponse(
            content="<h2>Web UI disabled. Set ENABLE_WEB_INTERFACE=true to enable.</h2>",
            status_code=200,
        )
    html = _build_web_ui()
    return HTMLResponse(content=html)


def _build_web_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OpenSecOpsEnv – Debug UI</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1   { color: #58a6ff; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-bottom: 12px; }
  button { background: #238636; color: #fff; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
  button:hover { background: #2ea043; }
  input, select { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 8px; border-radius: 4px; }
  pre  { white-space: pre-wrap; word-break: break-word; font-size: 12px; }
  .reward { color: #3fb950; font-weight: bold; }
  .penalty { color: #f85149; font-weight: bold; }
</style>
</head>
<body>
<h1>🔐 OpenSecOpsEnv – Interactive Debug</h1>

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

<div class="panel">
  <h3>Last Result</h3>
  <pre id="result">—</pre>
</div>

<div class="panel">
  <button onclick="doState()">Fetch Full State</button>
  <button onclick="doGrade()">Grade Episode</button>
  <pre id="state_out">—</pre>
</div>

<script>
async function doReset() {
  const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: document.getElementById('task_id').value})});
  const d = await r.json();
  document.getElementById('result').textContent = JSON.stringify(d, null, 2);
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
  const r = await fetch('/state');
  const d = await r.json();
  document.getElementById('state_out').textContent = JSON.stringify(d, null, 2);
}
async function doGrade() {
  const r = await fetch('/grade', {method:'POST'});
  const d = await r.json();
  document.getElementById('state_out').textContent = JSON.stringify(d, null, 2);
}
</script>
</body>
</html>"""
