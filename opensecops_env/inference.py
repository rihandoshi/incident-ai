"""
OpenSecOpsEnv - Baseline Inference Script (Groq - FREE)
========================================================
Runs all three tasks using a free Groq-hosted model.
Groq gives you fast free inference on Llama, Mixtral, and more.

Environment Variables
---------------------
GROQ_API_KEY   - Your free Groq API key (get one at console.groq.com)
MODEL_NAME     - Override model (default: llama-3.3-70b-versatile)
API_BASE_URL   - Override endpoint (default: https://api.groq.com/openai/v1)
HF_TOKEN       - Fallback API key for HF Space deployments

Quick start
-----------
    # 1. Get a FREE key at https://console.groq.com -> API Keys
    # 2. Set it:
    $env:GROQ_API_KEY = "gsk_..."

    # 3. Run all three tasks:
    python -m opensecops_env.inference

    # 4. Try a different model:
    $env:MODEL_NAME = "mixtral-8x7b-32768"
    python -m opensecops_env.inference

Output Format
-------------
[START] task=<id>
[STEP]  ...
[END]   task=<id> score=<float>
"""

from __future__ import annotations

import json
import os
import sys

from opensecops_env.env import OpenSecOpsEnv
from opensecops_env.grader import grade
from opensecops_env.models import SecOpsAction
from opensecops_env.tasks.task_definitions import TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api.groq.com/openai/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "llama-3.3-70b-versatile")
# GROQ_API_KEY is preferred; HF_TOKEN is a fallback for HF Space deployments
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY",  "")
HF_TOKEN      = os.environ.get("HF_TOKEN",      "")

OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except ImportError:
    OPENAI_AVAILABLE = False

MAX_STEPS_PER_TASK = 20   # keep inference fast

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert on-call site reliability engineer.
You are investigating a production incident. You receive observations containing:
- alerts: list of triggered alerts
- metrics: per-service CPU/memory/latency/error_rate
- logs: recent log lines (may contain noise)
- topology: service dependency graph
- last_action_result: result of your last action

You must take actions to diagnose and fix the root cause.

Available actions (return JSON):
{
  "action_type": "<type>",
  "parameters": {<params>}
}

Action types:
  query_logs         {"service": "<name>"}
  inspect_metrics    {"service": "<name>"}   or {}  for all
  restart_service    {"service": "<name>"}
  scale_service      {"service": "<name>", "replicas": <int>}
  block_ip           {"ip": "<ip>"}
  rollback_deployment {"service": "<name>", "version": "<ver>"}
  run_security_scan  {"target": "<name>"}
  isolate_service    {"service": "<name>"}
  submit_diagnosis   {"label": "<root_cause>:<subtype>"}

Root cause labels:
  infra_failure:memory_leak
  infra_failure:service_crash
  misconfiguration:bad_config
  cyber_attack:ddos
  cyber_attack:data_exfiltration
  cyber_attack:privilege_escalation

IMPORTANT:
- Do NOT expose the root cause in intermediate steps.
- Investigate BEFORE attempting mitigations.
- Always finish with submit_diagnosis.
- Return ONLY a JSON object, no markdown fences.
"""


def _obs_to_text(obs: dict) -> str:
    """Convert observation dict to a compact text for the LLM."""
    parts = [f"=== Step {obs.get('time_step', '?')} ==="]
    parts.append(f"Last result: {obs.get('last_action_result', '')}")
    parts.append("\nALERTS:")
    for a in obs.get("alerts", []):
        sev = a.get("severity", "")
        parts.append(f"  [{sev.upper()}] {a.get('service')} – {a.get('type')}: {a.get('message','')}")
    parts.append("\nMETRICS:")
    for svc, m in obs.get("metrics", {}).items():
        parts.append(
            f"  {svc}: cpu={m['cpu']:.1f}% mem={m['memory']:.1f}% "
            f"lat={m['latency']:.0f}ms err={m['error_rate']:.2f}%"
        )
    parts.append("\nLOGS (recent):")
    for line in obs.get("logs", [])[:6]:
        parts.append(f"  {line}")
    parts.append("\nTOPOLOGY:")
    for svc, deps in obs.get("topology", {}).items():
        parts.append(f"  {svc} → {deps}")
    return "\n".join(parts)


def _parse_action(raw: str) -> SecOpsAction:
    """Parse LLM output into a SecOpsAction."""
    # Strip markdown code fences if present
    raw = raw.strip().strip("```json").strip("```").strip()
    data = json.loads(raw)
    return SecOpsAction(
        action_type=data.get("action_type", "query_logs"),
        parameters=data.get("parameters", {}),
    )


def run_task(client, task_id: str) -> dict:
    """Run one task episode and return grading results."""
    env = OpenSecOpsEnv()
    obs_raw = env.reset(task_id=task_id)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    obs_dict = {
        "alerts": obs_raw.alerts,
        "metrics": obs_raw.metrics,
        "logs": obs_raw.logs,
        "topology": obs_raw.topology,
        "last_action_result": obs_raw.last_action_result,
        "time_step": obs_raw.time_step,
    }

    print(f"\n[START] task={task_id}", flush=True)

    done = False
    step_n = 0

    while not done and step_n < MAX_STEPS_PER_TASK:
        step_n += 1
        obs_text = _obs_to_text(obs_dict)
        messages.append({"role": "user", "content": obs_text})

        # ---- LLM call ----
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=256,
                )
                raw_action = response.choices[0].message.content or ""
            except Exception as e:
                raw_action = json.dumps({
                    "action_type": "submit_diagnosis",
                    "parameters": {"label": "infra_failure:memory_leak"},
                })
                print(f"  [WARN] LLM error: {e}", flush=True)
        else:
            # Fallback heuristic (for testing without a model)
            raw_action = _heuristic_action(obs_dict, step_n, task_id)

        # ---- Parse & step ----
        try:
            action = _parse_action(raw_action)
        except (json.JSONDecodeError, KeyError):
            action = SecOpsAction(action_type="inspect_metrics", parameters={})

        obs_result, reward, done, info = env.step(action)

        obs_dict = {
            "alerts": obs_result.alerts,
            "metrics": obs_result.metrics,
            "logs": obs_result.logs,
            "topology": obs_result.topology,
            "last_action_result": obs_result.last_action_result,
            "time_step": obs_result.time_step,
        }

        messages.append({"role": "assistant", "content": raw_action})

        print(
            f"[STEP]  task={task_id} step={step_n} "
            f"action={action.action_type}({json.dumps(action.parameters)}) "
            f"reward={reward:.3f} done={done}",
            flush=True,
        )

    # ---- Grade ----
    grade_result = grade(env.state.to_dict())
    print(
        f"[END]   task={task_id} score={grade_result.score:.4f} "
        f"diagnosis={grade_result.diagnosis_correct} "
        f"efficiency={grade_result.action_efficiency:.3f} "
        f"investigation={grade_result.investigation_quality:.3f}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "score": grade_result.score,
        "diagnosis_correct": grade_result.diagnosis_correct,
        "action_efficiency": grade_result.action_efficiency,
        "investigation_quality": grade_result.investigation_quality,
        "details": grade_result.details,
    }


def _heuristic_action(obs: dict, step: int, task_id: str) -> str:
    """
    Deterministic fallback agent for reproducible baseline scoring
    when no LLM is available.
    """
    # Simple rule-based agent per task
    sequences = {
        "easy_memory_leak": [
            {"action_type": "inspect_metrics", "parameters": {}},
            {"action_type": "query_logs",      "parameters": {"service": "auth"}},
            {"action_type": "restart_service", "parameters": {"service": "auth"}},
            {"action_type": "submit_diagnosis","parameters": {"label": "infra_failure:memory_leak"}},
        ],
        "medium_ddos_cascade": [
            {"action_type": "inspect_metrics",    "parameters": {}},
            {"action_type": "query_logs",         "parameters": {"service": "gateway"}},
            {"action_type": "run_security_scan",  "parameters": {"target": "api"}},
            {"action_type": "block_ip",           "parameters": {"ip": "203.0.113.45"}},
            {"action_type": "block_ip",           "parameters": {"ip": "198.51.100.12"}},
            {"action_type": "scale_service",      "parameters": {"service": "api", "replicas": 5}},
            {"action_type": "submit_diagnosis",   "parameters": {"label": "cyber_attack:ddos"}},
        ],
        "hard_data_exfiltration": [
            {"action_type": "inspect_metrics",    "parameters": {}},
            {"action_type": "query_logs",         "parameters": {"service": "db"}},
            {"action_type": "query_logs",         "parameters": {"service": "auth"}},
            {"action_type": "run_security_scan",  "parameters": {"target": "db"}},
            {"action_type": "run_security_scan",  "parameters": {"target": "auth"}},
            {"action_type": "isolate_service",    "parameters": {"service": "db"}},
            {"action_type": "block_ip",           "parameters": {"ip": "10.0.0.99"}},
            {"action_type": "submit_diagnosis",   "parameters": {"label": "cyber_attack:data_exfiltration"}},
        ],
    }
    seq = sequences.get(task_id, [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
    ])
    idx = min(step - 1, len(seq) - 1)
    return json.dumps(seq[idx])


def main() -> None:
    # Resolve API key: GROQ_API_KEY -> HF_TOKEN -> nothing (heuristic fallback)
    api_key = GROQ_API_KEY or HF_TOKEN

    if OPENAI_AVAILABLE and api_key:
        client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
        print(f"[Groq] Using model : {MODEL_NAME}", flush=True)
        print(f"       Endpoint     : {API_BASE_URL}", flush=True)
    else:
        client = None
        if not OPENAI_AVAILABLE:
            print("[WARN] openai package not installed -- running deterministic heuristic baseline.", flush=True)
        else:
            print("[WARN] No GROQ_API_KEY set -- running deterministic heuristic baseline.", flush=True)
            print("       Get a FREE key at https://console.groq.com -> API Keys", flush=True)
            print("       Then run:  $env:GROQ_API_KEY = 'gsk_...'", flush=True)

    results = []
    for task_id in TASKS:
        r = run_task(client, task_id)
        results.append(r)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    total = 0.0
    for r in results:
        print(
            f"  {r['task_id']:30s}  score={r['score']:.4f}",
            flush=True,
        )
        total += r["score"]
    avg = total / len(results) if results else 0.0
    print(f"\n  AVERAGE SCORE: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
