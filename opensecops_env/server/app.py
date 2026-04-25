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
GET  /dashboard      – 🔥 Live AI Demo Dashboard
GET  /demo/stream    – SSE stream: run a full episode step-by-step (for dashboard)

Session model
-------------
Each reset() call creates a named session keyed by `session_id`
(defaults to the task_id). This allows multiple concurrent agents
to run different tasks without colliding.
"""

from __future__ import annotations

import asyncio
import json
import os
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
    title="OpenSecOpsEnv",
    description=(
        "SecOps incident response environment for RL agent evaluation. "
        "Simulates memory leaks, DDoS attacks, misconfiguration, and "
        "data exfiltration scenarios across 4 tasks of increasing difficulty."
    ),
    version="0.3.0",
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
    return {"status": "ok", "environment": "opensecops", "version": "0.3.0"}


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
    sid = req.session_id
    if sid not in _ma_sessions:
        _ma_sessions[sid] = MultiAgentSecOpsEnv()
    
    env = _ma_sessions[sid]
    # If task_id is "auto", curriculum decides the next task
    if req.task_id == "auto":
        task_cfg = _curriculum.get_next_task()
        tid = task_cfg["task_id"]
        # Temporary registration if it's a generated task
        if tid not in TASKS:
            TASKS[tid] = task_cfg
    else:
        tid = req.task_id or "hard_data_exfiltration"
        
    env.reset(tid)
    return StateResponse(state=env.state)

@app.post("/multi/blue/step", response_model=MultiAgentStepResponse)
def multi_blue_step(req: StepRequest) -> MultiAgentStepResponse:
    env = _get_ma_session(req.session_id)
    action = SecOpsAction(
        action_type=req.action_type,
        parameters=req.parameters,
    )
    obs, reward, done, info = env.blue_step(action)
    
    if done:
        # Record final score in curriculum
        g = grade(env.state)
        _curriculum.record_score(env._ma_state.task_id, g.score)
        
    return MultiAgentStepResponse(
        observation=obs.to_dict(),
        reward=reward,
        done=done,
        info=info,
    )

@app.post("/multi/red/step", response_model=MultiAgentStepResponse)
def multi_red_step(req: RedActionRequest) -> MultiAgentStepResponse:
    env = _get_ma_session(req.session_id)
    
    # If action_type is missing or auto, use heuristic Red agent
    action = None
    if req.action_type and req.action_type != "auto":
        action = RedAction(
            action_type=req.action_type,
            parameters=req.parameters,
        )
        
    obs, reward, done, info = env.red_step(action)
    return MultiAgentStepResponse(
        observation=obs,  # red's limited view
        reward=reward,
        done=done,
        info=info,
    )

# ---------------------------------------------------------------------------
# Curriculum Endpoints
# ---------------------------------------------------------------------------

@app.get("/curriculum/summary", response_model=CurriculumSummaryResponse)
def curriculum_summary() -> CurriculumSummaryResponse:
    return CurriculumSummaryResponse(
        current_level=_curriculum.current_level,
        total_episodes=_curriculum.episode_count,
        summary_text=_curriculum.summary(),
        level_up_history=_curriculum.level_up_history,
    )


# ---------------------------------------------------------------------------
# SSE Demo Stream  — powers the live dashboard
# ---------------------------------------------------------------------------

# Heuristic playbooks (deterministic, perfect agent)
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


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/demo/stream")
async def demo_stream(
    task_id: str = "easy_memory_leak",
    mode: str = "trained",   # "trained" | "untrained"
    speed: float = 1.5,      # seconds between steps
):
    """
    Server-Sent Events stream of a full episode.
    Powers the live dashboard. mode=trained uses expert agent,
    mode=untrained uses the bad agent to show contrast.
    """
    async def event_gen():
        try:
            playbook = (
                _HEURISTIC_PLAYBOOKS if mode == "trained"
                else _BAD_AGENT_PLAYBOOKS
            ).get(task_id, _HEURISTIC_PLAYBOOKS.get(task_id, []))

            env = OpenSecOpsEnv()
            obs = env.reset(task_id)

            # Send initial state
            yield _sse({
                "type": "reset",
                "task_id": task_id,
                "mode": mode,
                "observation": {
                    "alerts": obs.alerts,
                    "metrics": obs.metrics,
                    "logs": obs.logs,
                    "topology": obs.topology,
                    "last_action_result": obs.last_action_result,
                    "time_step": obs.time_step,
                },
            })

            cumulative = 0.0
            rewards_history: list[float] = []

            for i, action_dict in enumerate(playbook):
                await asyncio.sleep(max(0.5, speed))

                action = SecOpsAction(
                    action_type=action_dict["action_type"],
                    parameters=action_dict.get("parameters", {}),
                )
                obs, reward, done, info = env.step(action)
                cumulative += reward
                rewards_history.append(round(reward, 4))

                yield _sse({
                    "type": "step",
                    "step": i + 1,
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "reward": round(reward, 4),
                    "cumulative_reward": round(cumulative, 4),
                    "rewards_history": rewards_history,
                    "done": done,
                    "observation": {
                        "alerts": obs.alerts,
                        "metrics": obs.metrics,
                        "logs": obs.logs,
                        "topology": obs.topology,
                        "last_action_result": obs.last_action_result,
                        "time_step": obs.time_step,
                    },
                })

                if done:
                    break

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

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Live AI Demo Dashboard — the main judge-facing demo."""
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
<h1>🔐 OpenSecOpsEnv – Debug UI</h1>
<p><a href="/dashboard" style="color:#58a6ff">→ Open Live Dashboard</a></p>
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
# Dashboard HTML (inline — no static files needed, works in Docker/HF Space)
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenSecOpsEnv — Live AI Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f8fafc;
    --bg2: #ffffff;
    --bg3: #f1f5f9;
    --border: #e2e8f0;
    --border2: #cbd5e1;
    --blue: #2563eb;
    --green: #16a34a;
    --red: #dc2626;
    --orange: #d97706;
    --yellow: #ca8a04;
    --purple: #7c3aed;
    --text: #0f172a;
    --text2: #64748b;
    --card-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --card-shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  .app { position: relative; }

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
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: var(--blue); }
  .logo span { color: var(--text2); font-weight: 400; }
  .badge {
    font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
    padding: 3px 8px; border-radius: 4px; text-transform: uppercase;
  }
  .badge-openenv { background: #eff6ff; color: var(--blue); border: 1px solid #bfdbfe; }
  .badge-live { background: #fef2f2; color: var(--red); border: 1px solid #fecaca; animation: pulse-badge 2s infinite; }
  @keyframes pulse-badge { 0%,100%{opacity:1} 50%{opacity:0.6} }
  .header-status { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse-dot 2s infinite; }
  @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.5} }

  /* ---- Controls bar ---- */
  .controls {
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid var(--border);
    background: #ffffff;
    flex-wrap: wrap;
  }
  .control-group { display: flex; flex-direction: column; gap: 3px; }
  .control-label { font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text2); }
  select, .btn {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    border-radius: 6px;
    border: 1px solid var(--border2);
    background: #ffffff;
    color: var(--text);
    padding: 7px 12px;
    cursor: pointer;
    outline: none;
    transition: border-color 0.15s;
  }
  select:hover, select:focus { border-color: var(--blue); }
  .btn { font-weight: 600; display: flex; align-items: center; gap: 6px; }
  .btn-primary { background: var(--blue); color: #fff; border-color: var(--blue); }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ghost { background: transparent; color: var(--text2); }
  .btn-ghost:hover { color: var(--text); border-color: var(--border2); background: var(--bg3); }
  .mode-toggle { display: flex; border-radius: 6px; overflow: hidden; border: 1px solid var(--border2); }
  .mode-btn { padding: 7px 14px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; background: #ffffff; color: var(--text2); border: none; }
  .mode-btn.active-trained { background: #f0fdf4; color: var(--green); }
  .mode-btn.active-untrained { background: #fef2f2; color: var(--red); }

  /* ---- Main layout ---- */
  .main { display: grid; grid-template-columns: 340px 1fr 300px; gap: 0; height: calc(100vh - 120px); }

  /* ---- Panel base ---- */
  .panel {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }
  .panel-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text2);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    background: #ffffff;
  }
  .panel-header-dot { width: 8px; height: 8px; border-radius: 50%; }
  .panel-body { flex: 1; overflow-y: auto; padding: 14px; }
  .panel-body::-webkit-scrollbar { width: 4px; }
  .panel-body::-webkit-scrollbar-track { background: transparent; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* ---- Left Panel: Scenario + Metrics + Topology ---- */
  .section-title {
    font-size: 10px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--text2); margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
  }
  .section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  .scenario-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
  }
  .scenario-name { font-size: 13px; font-weight: 600; color: var(--blue); margin-bottom: 4px; }
  .scenario-desc { font-size: 11px; color: var(--text2); line-height: 1.5; }
  .difficulty-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; font-weight: 600; border-radius: 4px; padding: 2px 7px; margin-top: 6px;
  }
  .diff-easy { background: #f0fdf4; color: var(--green); border: 1px solid #bbf7d0; }
  .diff-medium { background: #fefce8; color: var(--yellow); border: 1px solid #fef08a; }
  .diff-medium_hard { background: #fff7ed; color: var(--orange); border: 1px solid #fed7aa; }
  .diff-hard { background: #fef2f2; color: var(--red); border: 1px solid #fecaca; }

  /* Metrics */
  .metric-row {
    display: grid; grid-template-columns: 80px 1fr 45px;
    align-items: center; gap: 8px; margin-bottom: 8px;
  }
  .metric-svc { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text); font-weight: 500; }
  .metric-bars { display: flex; flex-direction: column; gap: 2px; }
  .metric-bar-wrap { display: flex; align-items: center; gap: 4px; }
  .metric-bar-label { font-size: 9px; color: var(--text2); width: 24px; }
  .metric-bar-bg { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .metric-bar-fill { height: 100%; border-radius: 2px; transition: width 0.6s cubic-bezier(0.4,0,0.2,1), background 0.6s; }
  .bar-ok { background: var(--green); }
  .bar-warn { background: var(--orange); }
  .bar-crit { background: var(--red); }
  .metric-err { font-family: 'JetBrains Mono', monospace; font-size: 10px; text-align: right; }

  /* Alerts */
  .alert-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 8px 10px; border-radius: 6px; margin-bottom: 6px;
    border-left: 3px solid;
    font-size: 11px; animation: slide-in 0.25s ease;
  }
  @keyframes slide-in { from { opacity:0; transform: translateX(-6px); } to { opacity:1; transform: translateX(0); } }
  .alert-critical { background: #fef2f2; border-color: var(--red); color: #991b1b; }
  .alert-warning { background: #fff7ed; border-color: var(--orange); color: #92400e; }
  .alert-info { background: #eff6ff; border-color: var(--blue); color: #1e40af; }
  .alert-sev { font-size: 9px; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 2px; }
  .alert-msg { font-size: 11px; line-height: 1.4; }
  .alert-svc { font-family: 'JetBrains Mono', monospace; font-weight: 600; }

  /* Topology */
  .topology-graph { display: flex; flex-direction: column; gap: 6px; }
  .topo-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .topo-node {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500;
    padding: 4px 10px; border-radius: 5px; background: #f1f5f9; border: 1px solid var(--border2);
    color: var(--text); transition: all 0.3s;
  }
  .topo-node.affected { border-color: var(--red); color: var(--red); background: #fef2f2; }
  .topo-node.mitigated { border-color: var(--green); color: var(--green); background: #f0fdf4; }
  .topo-arrow { color: var(--text2); font-size: 11px; }

  /* ---- Center Panel: Action Feed ---- */
  .center-panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--bg); }
  .action-feed { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 8px; }
  .action-feed::-webkit-scrollbar { width: 4px; }
  .action-feed::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  .action-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 14px;
    box-shadow: var(--card-shadow);
    animation: card-in 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  @keyframes card-in { from { opacity:0; transform: translateY(10px) scale(0.98); } to { opacity:1; transform: translateY(0) scale(1); } }
  .action-card.positive { border-left: 3px solid var(--green); }
  .action-card.negative { border-left: 3px solid var(--red); }
  .action-card.neutral  { border-left: 3px solid var(--border2); }
  .action-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; }
  .action-step { font-size: 10px; color: var(--text2); font-weight: 600; }
  .action-type {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500;
    padding: 2px 7px; border-radius: 4px;
  }
  .action-type.mitigation { background: #f5f3ff; color: var(--purple); }
  .action-type.investigation { background: #eff6ff; color: var(--blue); }
  .action-type.terminal { background: #fefce8; color: var(--yellow); }
  .action-params { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text2); margin-bottom: 5px; }
  .action-result { font-size: 12px; color: var(--text); line-height: 1.5; }
  .reward-badge {
    display: inline-flex; align-items: center; gap: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px;
  }
  .reward-pos { color: var(--green); background: #f0fdf4; }
  .reward-neg { color: var(--red); background: #fef2f2; }
  .reward-zero { color: var(--text2); background: var(--bg3); }

  .thinking-indicator {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; border-radius: 8px;
    background: #f8fafc; border: 1px dashed var(--border2);
    font-size: 12px; color: var(--text2); animation: thinking-pulse 1.5s infinite;
  }
  @keyframes thinking-pulse { 0%,100%{opacity:0.7} 50%{opacity:1} }
  .thinking-dots span {
    display: inline-block; width: 5px; height: 5px; border-radius: 50%;
    background: var(--blue); margin: 0 2px;
    animation: dot-bounce 1.2s infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dot-bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }

  /* Log stream */
  .log-stream {
    border-top: 1px solid var(--border);
    height: 130px;
    overflow-y: auto;
    padding: 8px 14px;
    flex-shrink: 0;
    background: #1e293b;
  }
  .log-stream::-webkit-scrollbar { width: 3px; }
  .log-stream::-webkit-scrollbar-thumb { background: #334155; }
  .log-line { font-family: 'JetBrains Mono', monospace; font-size: 10px; line-height: 1.7; }
  .log-error { color: #f87171; }
  .log-warn  { color: #fbbf24; }
  .log-crit  { color: #fb923c; font-weight: 500; }
  .log-info  { color: #94a3b8; }

  /* ---- Right Panel ---- */
  .right-panel { display: flex; flex-direction: column; background: var(--bg); }
  .chart-wrap { padding: 12px 14px; border-bottom: 1px solid var(--border); background: #ffffff; }
  .score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 12px 14px; border-bottom: 1px solid var(--border); background: #ffffff; }
  .score-card {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; text-align: center;
    box-shadow: var(--card-shadow);
  }
  .score-val { font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .score-lbl { font-size: 9px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text2); margin-top: 2px; }
  .score-big { grid-column: 1 / -1; background: #eff6ff; border-color: #bfdbfe; }

  .progress-wrap { padding: 12px 14px; background: #ffffff; flex: 1; }
  .progress-item { margin-bottom: 12px; }
  .progress-header { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px; }
  .progress-name { color: var(--text2); }
  .progress-val { font-family: 'JetBrains Mono', monospace; color: var(--text); font-weight: 600; }
  .progress-bar-bg { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .progress-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }

  /* Episode end overlay */
  .episode-end {
    display: none; position: fixed; inset: 0; z-index: 200;
    background: rgba(248,250,252,0.92); backdrop-filter: blur(8px);
    align-items: center; justify-content: center;
  }
  .episode-end.show { display: flex; }
  .end-card {
    background: #ffffff; border: 1px solid var(--border);
    border-radius: 12px; padding: 36px; text-align: center;
    max-width: 480px; width: 90%;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    animation: card-in 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  .end-icon { font-size: 44px; margin-bottom: 10px; }
  .end-title { font-size: 22px; font-weight: 700; margin-bottom: 6px; color: var(--text); }
  .end-score { font-size: 52px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-bottom: 14px; }
  .end-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
  .end-col { padding: 12px; border-radius: 8px; }
  .end-col-trained { background: #f0fdf4; border: 1px solid #bbf7d0; }
  .end-col-untrained { background: #fef2f2; border: 1px solid #fecaca; }
  .end-col-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
  .end-col-score { font-size: 26px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .end-col-trained .end-col-title { color: var(--green); }
  .end-col-trained .end-col-score { color: var(--green); }
  .end-col-untrained .end-col-title { color: var(--red); }
  .end-col-untrained .end-col-score { color: var(--red); }

  /* Info toast (replaces alert()) */
  .toast {
    position: fixed; bottom: 24px; right: 24px; z-index: 300;
    background: var(--text); color: #ffffff; border-radius: 8px;
    padding: 12px 18px; font-size: 13px; font-weight: 500;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    animation: toast-in 0.3s ease; max-width: 360px;
  }
  @keyframes toast-in { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: translateY(0); } }

  /* Misc */
  .empty-state { color: var(--text2); font-size: 12px; text-align: center; padding: 36px 20px; line-height: 1.7; }
  .tag { font-size: 10px; padding: 2px 7px; border-radius: 4px; font-weight: 600; }

  @media (max-width: 1100px) {
    .main { grid-template-columns: 260px 1fr 240px; }
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
    <span class="badge badge-live" id="liveBadge" style="display:none">● Live</span>
  </div>
  <div class="header-status">
    <div class="status-dot"></div>
    <span>Server online</span>
  </div>
</div>

<!-- Controls -->
<div class="controls">
  <div class="control-group">
    <span class="control-label">Scenario</span>
    <select id="scenarioSelect">
      <option value="easy_memory_leak">🟢 Memory Leak — Easy</option>
      <option value="medium_ddos_cascade">🟡 DDoS Cascade — Medium</option>
      <option value="medium_hard_bad_deployment">🟠 Bad Deployment — Medium-Hard</option>
      <option value="hard_data_exfiltration" selected>🔴 Data Exfiltration — Hard</option>
    </select>
  </div>

  <div class="control-group">
    <span class="control-label">Agent Mode</span>
    <div class="mode-toggle">
      <button class="mode-btn active-trained" id="modeTrainedBtn" onclick="setMode('trained')">✦ Trained AI</button>
      <button class="mode-btn" id="modeUntrainedBtn" onclick="setMode('untrained')">✗ Untrained</button>
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

  <button class="btn btn-primary" id="startBtn" onclick="startDemo()">
    ▶ Run Episode
  </button>
  <button class="btn btn-ghost" onclick="resetUI()">↺ Reset</button>
  <button class="btn btn-ghost" onclick="showComparison()">⇄ Compare Agents</button>
</div>

<!-- Main -->
<div class="main">

  <!-- LEFT: Scenario info + Metrics + Alerts + Topology -->
  <div class="panel">
    <div class="panel-header">
      <span>System State</span>
      <div class="panel-header-dot" style="background:var(--blue)"></div>
    </div>
    <div class="panel-body" id="leftPanel">
      <div class="empty-state">
        Select a scenario and click<br>
        <strong>Run Episode</strong> to start the demo.
      </div>
    </div>
  </div>

  <!-- CENTER: Action feed + Log stream -->
  <div class="center-panel">
    <div class="panel-header">
      <span>Agent Action Feed</span>
      <div class="panel-header-dot" style="background:var(--purple)"></div>
    </div>
    <div class="action-feed" id="actionFeed">
      <div class="empty-state">
        🤖 The agent will appear here<br>as it investigates the incident.
      </div>
    </div>
    <div class="log-stream" id="logStream">
      <div class="log-line log-info">// System logs will stream here during the episode...</div>
    </div>
  </div>

  <!-- RIGHT: Chart + Scores -->
  <div class="right-panel">
    <div class="panel-header">
      <span>Reward Curve</span>
      <div class="panel-header-dot" style="background:var(--green)"></div>
    </div>
    <div class="chart-wrap">
      <canvas id="rewardChart" height="160"></canvas>
    </div>
    <div class="score-grid" id="scoreGrid">
      <div class="score-card score-big">
        <div class="score-val" id="scoreTotal" style="color:var(--text2)">—</div>
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

<!-- Episode End Modal -->
<div class="episode-end" id="episodeEnd">
  <div class="end-card">
    <div class="end-icon" id="endIcon">🎯</div>
    <div class="end-title" id="endTitle">Episode Complete</div>
    <div class="end-score" id="endScore">—</div>
    <div class="end-comparison" id="endComparison" style="display:none">
      <div class="end-col end-col-trained">
        <div class="end-col-title">✦ Trained AI</div>
        <div class="end-col-score" id="cmpTrained">—</div>
      </div>
      <div class="end-col end-col-untrained">
        <div class="end-col-title">✗ Untrained</div>
        <div class="end-col-score" id="cmpUntrained">—</div>
      </div>
    </div>
    <button class="btn btn-primary" onclick="closeEnd()" style="margin:0 auto;display:flex">
      ↺ Run Again
    </button>
  </div>
</div>

</div><!-- /app -->

<script>
// ═══════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════
let currentMode = 'trained';
let currentTask = 'hard_data_exfiltration';
let sse = null;
let rewardChart = null;
let steps = 0;
let comparisonScores = {};

const TASK_META = {
  'easy_memory_leak': {
    name: 'Memory Leak — Auth Service',
    diff: 'easy',
    desc: 'The auth service has a progressive memory leak. Metrics show rising memory usage. The agent must investigate and restart the correct service.',
  },
  'medium_ddos_cascade': {
    name: 'DDoS Cascade Attack',
    diff: 'medium',
    desc: 'A DDoS attack from two IP ranges cascades through gateway → api → auth. The agent must correlate logs, block attacking IPs, and scale the API.',
  },
  'medium_hard_bad_deployment': {
    name: 'Bad Deployment — Redis Misconfiguration',
    diff: 'medium_hard',
    desc: 'api v2.4.1 pushed an invalid Redis connection string, sending cache into a reconnect storm. The agent must rollback the deployment.',
  },
  'hard_data_exfiltration': {
    name: 'Data Exfiltration (Disguised)',
    diff: 'hard',
    desc: 'A compromised service account "reports_bot" is exfiltrating 4GB+ of data. 55% noise level with a false cache alert to mislead the responder.',
  },
};

const ACTION_CLASS = {
  query_logs: 'investigation', inspect_metrics: 'investigation', run_security_scan: 'investigation',
  restart_service: 'mitigation', scale_service: 'mitigation', block_ip: 'mitigation',
  rollback_deployment: 'mitigation', isolate_service: 'mitigation',
  submit_diagnosis: 'terminal',
};

const ACTION_ICON = {
  query_logs: '📋',inspect_metrics: '📊',run_security_scan: '🔍',
  restart_service: '🔄',scale_service: '⚖️',block_ip: '🚫',
  rollback_deployment: '⏪',isolate_service: '🔒',submit_diagnosis: '🎯',
};

// ═══════════════════════════════════════════════════════
// Chart init
// ═══════════════════════════════════════════════════════
function initChart() {
  const ctx = document.getElementById('rewardChart').getContext('2d');
  if (rewardChart) rewardChart.destroy();
  rewardChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Step Reward',
        data: [],
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37,99,235,0.08)',
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: '#2563eb',
        tension: 0.3,
        fill: true,
      }, {
        label: 'Cumulative',
        data: [],
        borderColor: '#16a34a',
        borderWidth: 1.5,
        borderDash: [4, 3],
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: {
          labels: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' }, boxWidth: 12 }
        },
        tooltip: { backgroundColor: '#ffffff', titleColor: '#0f172a', bodyColor: '#64748b', borderColor: '#e2e8f0', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(226,232,240,0.8)' } },
        y: {
          ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } },
          grid: { color: 'rgba(226,232,240,0.8)' },
          min: -1.2, max: 1.2,
        }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════
// UI helpers
// ═══════════════════════════════════════════════════════
function setMode(m) {
  currentMode = m;
  document.getElementById('modeTrainedBtn').className = 'mode-btn' + (m==='trained' ? ' active-trained' : '');
  document.getElementById('modeUntrainedBtn').className = 'mode-btn' + (m==='untrained' ? ' active-untrained' : '');
}

function resetUI() {
  if (sse) { sse.close(); sse = null; }
  document.getElementById('actionFeed').innerHTML = '<div class="empty-state">🤖 The agent will appear here<br>as it investigates the incident.</div>';
  document.getElementById('logStream').innerHTML = '<div class="log-line log-info">// System logs will stream here during the episode...</div>';
  document.getElementById('leftPanel').innerHTML = '<div class="empty-state">Select a scenario and click<br><strong>Run Episode</strong> to start the demo.</div>';
  document.getElementById('scoreTotal').textContent = '—';
  document.getElementById('scoreTotal').style.color = 'var(--text2)';
  document.getElementById('scoreCumReward').textContent = '0.00';
  document.getElementById('scoreSteps').textContent = '0';
  ['pDiagnosis','pEfficiency','pInvestigation'].forEach(id => document.getElementById(id).textContent = '—');
  ['pDiagnosisBar','pEfficiencyBar','pInvestigationBar'].forEach(id => {
    const el = document.getElementById(id);
    el.style.width = '0%';
  });
  document.getElementById('liveBadge').style.display = 'none';
  document.getElementById('startBtn').disabled = false;
  initChart();
  steps = 0;
  // Note: we intentionally preserve comparisonScores so Compare still works after reset
}

function barClass(val, warn, crit) {
  if (val >= crit) return 'bar-crit';
  if (val >= warn) return 'bar-warn';
  return 'bar-ok';
}

function renderLeftPanel(obs, taskId) {
  const meta = TASK_META[taskId] || {};
  const diff = meta.diff || 'easy';
  // Track which services are affected from known task metadata (for topology highlighting)
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

  // Alerts
  if (obs.alerts && obs.alerts.length) {
    html += `<div class="section-title">Active Alerts</div>`;
    obs.alerts.slice(0, 6).forEach(a => {
      const cls = a.severity === 'critical' ? 'alert-critical' : a.severity === 'warning' ? 'alert-warning' : 'alert-info';
      html += `<div class="alert-item ${cls}">
        <div>
          <div class="alert-sev">[${(a.severity||'info').toUpperCase()}]</div>
          <div class="alert-msg"><span class="alert-svc">${a.service}</span> · ${a.message || a.type}</div>
        </div></div>`;
    });
  }

  // Metrics
  if (obs.metrics && Object.keys(obs.metrics).length) {
    html += `<div class="section-title" style="margin-top:12px">Service Metrics</div>`;
    Object.entries(obs.metrics).forEach(([svc, m]) => {
      const cpuCls = barClass(m.cpu, 60, 85);
      const memCls = barClass(m.memory, 70, 85);
      const latCls = m.latency > 500 ? 'bar-crit' : m.latency > 200 ? 'bar-warn' : 'bar-ok';
      const errColor = m.error_rate > 10 ? 'var(--red)' : m.error_rate > 5 ? 'var(--orange)' : 'var(--text2)';
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

  // Topology — with affected/mitigated node highlighting
  if (obs.topology && Object.keys(obs.topology).length) {
    html += `<div class="section-title" style="margin-top:12px">Service Topology</div><div class="topology-graph">`;
    Object.entries(obs.topology).forEach(([svc, deps]) => {
      const nodeClass = affected.has(svc) ? 'affected' : '';
      html += `<div class="topo-row"><span class="topo-node ${nodeClass}">${svc}</span>`;
      if (deps.length) {
        html += `<span class="topo-arrow">→</span>`;
        deps.forEach(d => {
          const dClass = affected.has(d) ? 'affected' : '';
          html += `<span class="topo-node ${dClass}">${d}</span>`;
        });
      }
      html += `</div>`;
    });
    html += `</div>`;
  }

  document.getElementById('leftPanel').innerHTML = html;
}

function appendAction(data) {
  const feed = document.getElementById('actionFeed');
  const isFirst = feed.querySelector('.empty-state');
  if (isFirst) feed.innerHTML = '';

  const r = data.reward;
  const rClass = r > 0 ? 'positive' : r < 0 ? 'negative' : 'neutral';
  const rBadgeClass = r > 0 ? 'reward-pos' : r < 0 ? 'reward-neg' : 'reward-zero';
  const rSign = r > 0 ? '+' : '';
  const aClass = ACTION_CLASS[data.action_type] || 'investigation';
  const icon = ACTION_ICON[data.action_type] || '⚙️';
  const params = Object.keys(data.parameters || {}).length ?
    Object.entries(data.parameters).map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ') : 'none';

  const card = document.createElement('div');
  card.className = `action-card ${rClass}`;
  card.innerHTML = `
    <div class="action-header">
      <span class="action-step">Step ${data.step}</span>
      <span class="action-type ${aClass}">${icon} ${data.action_type}</span>
      <span class="reward-badge ${rBadgeClass}">${rSign}${r.toFixed(2)}</span>
    </div>
    <div class="action-params">params: ${params}</div>
    <div class="action-result">${data.observation?.last_action_result || ''}</div>
  `;
  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
}

function appendLogs(logs) {
  if (!logs || !logs.length) return;
  const stream = document.getElementById('logStream');
  logs.forEach(line => {
    const div = document.createElement('div');
    const cls = line.includes('CRIT') ? 'log-crit' : line.includes('ERROR') ? 'log-error' : line.includes('WARN') ? 'log-warn' : 'log-info';
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
  // running cumulative
  let cum = 0;
  rewardChart.data.datasets[1].data = rewardsHistory.map(r => { cum += r; return Math.round(cum * 100) / 100; });
  rewardChart.update('none');
}

function updateScores(data) {
  document.getElementById('scoreCumReward').textContent = (data.cumulative_reward || 0).toFixed(2);
  document.getElementById('scoreSteps').textContent = data.step || steps;
}

function showThinking() {
  const feed = document.getElementById('actionFeed');
  const t = document.createElement('div');
  t.className = 'thinking-indicator';
  t.id = 'thinkingIndicator';
  t.innerHTML = `🤖 Agent processing...&nbsp;<span class="thinking-dots"><span></span><span></span><span></span></span>`;
  feed.appendChild(t);
  feed.scrollTop = feed.scrollHeight;
}

function hideThinking() {
  const t = document.getElementById('thinkingIndicator');
  if (t) t.remove();
}

function showEpisodeEnd(gradeData) {
  const score = gradeData.score;
  const correct = gradeData.diagnosis_correct > 0.9;
  document.getElementById('endIcon').textContent = correct ? '🏆' : score > 0.5 ? '✅' : '⚠️';
  document.getElementById('endTitle').textContent = correct ? 'Incident Resolved!' : score > 0.5 ? 'Partially Resolved' : 'Episode Failed';
  const scoreEl = document.getElementById('endScore');
  scoreEl.textContent = score.toFixed(3);
  scoreEl.style.color = score > 0.7 ? 'var(--green)' : score > 0.4 ? 'var(--yellow)' : 'var(--red)';

  if (comparisonScores.trained && comparisonScores.untrained) {
    document.getElementById('endComparison').style.display = 'grid';
    document.getElementById('cmpTrained').textContent = comparisonScores.trained.toFixed(3);
    document.getElementById('cmpUntrained').textContent = comparisonScores.untrained.toFixed(3);
  }

  document.getElementById('episodeEnd').classList.add('show');
}

function closeEnd() {
  document.getElementById('episodeEnd').classList.remove('show');
  resetUI();
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

// ═══════════════════════════════════════════════════════
// Main: run episode via SSE
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
      renderLeftPanel(data.observation, data.task_id);
      showThinking();
    }

    else if (data.type === 'step') {
      hideThinking();
      steps = data.step;
      appendAction(data);
      appendLogs(data.observation?.logs);
      renderLeftPanel(data.observation, currentTask);
      updateChart(data.rewards_history, data.cumulative_reward);
      updateScores(data);
      if (!data.done) showThinking();
    }

    else if (data.type === 'grade') {
      hideThinking();
      document.getElementById('liveBadge').style.display = 'none';
      document.getElementById('startBtn').disabled = false;
      updateChart(data.rewards_history, data.cumulative_reward);
      updateGradeUI(data);
      // Store for comparison
      comparisonScores[currentMode] = data.score;
      setTimeout(() => showEpisodeEnd(data), 800);
      sse.close();
    }

    else if (data.type === 'error') {
      hideThinking();
      document.getElementById('startBtn').disabled = false;
      document.getElementById('liveBadge').style.display = 'none';
      const feed = document.getElementById('actionFeed');
      feed.innerHTML += `<div class="alert-item alert-critical"><div><div class="alert-sev">[ERROR]</div><div>${data.message}</div></div></div>`;
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
    showToast('Run both "Trained AI" and "Untrained" modes first, then click Compare to see the difference!');
    return;
  }
  document.getElementById('endComparison').style.display = 'grid';
  document.getElementById('cmpTrained').textContent = (comparisonScores.trained || 0).toFixed(3);
  document.getElementById('cmpUntrained').textContent = (comparisonScores.untrained || 0).toFixed(3);
  const diff = (comparisonScores.trained || 0) - (comparisonScores.untrained || 0);
  document.getElementById('endIcon').textContent = '⇄';
  document.getElementById('endTitle').textContent = `Trained vs Untrained Agent`;
  document.getElementById('endScore').textContent = `+${(diff * 100).toFixed(0)}% better`;
  document.getElementById('endScore').style.color = 'var(--green)';
  document.getElementById('episodeEnd').classList.add('show');
}

// ═══════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  document.getElementById('scenarioSelect').addEventListener('change', e => { currentTask = e.target.value; });
});
</script>
</body>
</html>
"""
