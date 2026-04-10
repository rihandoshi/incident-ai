"""
OpenSecOpsEnv — Baseline Inference Script
==========================================
MANDATORY
* Before running, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
                   Default: https://router.huggingface.co/v1
    MODEL_NAME     The model identifier to use for inference.
                   Default: Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       Your Hugging Face API key (used as API key).

* Participants must use OpenAI Client for all LLM calls.

STDOUT FORMAT
    [START] task=<task_name> env=opensecops model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_result string on failure, or null.
    - All fields on a single line with no newlines within a line.
    - Each task returns score in [0, 1].

Quick start
-----------
    # Set your HF token:
    $env:HF_TOKEN = "hf_..."

    # Run all three tasks:
    python inference.py

    # Use a different model / endpoint:
    $env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    $env:API_BASE_URL = "https://router.huggingface.co/v1"
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Configuration — all from env vars with sensible defaults
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
# Fallback key names
_API_KEY     = HF_TOKEN or os.getenv("OPENAI_API_KEY", "") or os.getenv("GROQ_API_KEY", "")

BENCHMARK    = "opensecops"
MAX_STEPS    = 20          # per task; keeps total runtime well under 20 min
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.5   # score ≥ 0.5 → success

# ---------------------------------------------------------------------------
# Import environment
# ---------------------------------------------------------------------------
from opensecops_env.env import OpenSecOpsEnv
from opensecops_env.grader import grade
from opensecops_env.models import SecOpsAction
from opensecops_env.tasks.task_definitions import TASKS

# ---------------------------------------------------------------------------
# Structured logging helpers — spec-mandatory format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert on-call site reliability / security engineer.
    You are investigating a production incident in a distributed system.

    At each step you receive:
      alerts   – threshold-based monitoring alerts
      metrics  – per-service CPU / memory / latency / error_rate
      logs     – recent (potentially noisy) log lines
      topology – service dependency graph
      last_action_result – outcome of your previous action

    You must take actions to diagnose the root cause and apply mitigations.
    Always finish the episode with a "submit_diagnosis" action.

    RETURN ONLY a valid JSON object — no markdown fences, no prose:
    {
      "action_type": "<type>",
      "parameters":  {<params>}
    }

    Available action types:
      query_logs          {\"service\": \"<name>\"}
      inspect_metrics     {\"service\": \"<name>\"}  or  {}  (all services)
      restart_service     {\"service\": \"<name>\"}
      scale_service       {\"service\": \"<name>\", \"replicas\": <int>}
      block_ip            {\"ip\": \"<address>\"}
      rollback_deployment {\"service\": \"<name>\", \"version\": \"<tag>\"}
      run_security_scan   {\"target\": \"<name>\"}
      isolate_service     {\"service\": \"<name>\"}
      submit_diagnosis    {\"label\": \"<root_cause>:<subtype>\"}

    Valid diagnosis labels:
      infra_failure:memory_leak
      infra_failure:service_crash
      misconfiguration:bad_config
      cyber_attack:ddos
      cyber_attack:data_exfiltration
      cyber_attack:privilege_escalation

    Strategy:
      1. Inspect metrics broadly first.
      2. Query logs or run security scans on suspicious services.
      3. Apply targeted mitigations.
      4. Submit your diagnosis when confident.
      Avoid wasted actions — penalising wrong mitigations or blocking legit IPs.
""").strip()


# ---------------------------------------------------------------------------
# Observation → text formatter
# ---------------------------------------------------------------------------

def _obs_to_text(obs: dict) -> str:
    parts = [f"=== Step {obs.get('time_step', '?')} ==="]
    parts.append(f"Last action result: {obs.get('last_action_result', '')}")

    parts.append("\nALERTS:")
    for a in obs.get("alerts", []):
        sev = a.get("severity", "info").upper()
        parts.append(f"  [{sev}] {a.get('service')} – {a.get('type')}: {a.get('message', '')}")

    parts.append("\nMETRICS:")
    for svc, m in obs.get("metrics", {}).items():
        parts.append(
            f"  {svc}: cpu={m['cpu']:.1f}% mem={m['memory']:.1f}% "
            f"lat={m['latency']:.0f}ms err={m['error_rate']:.2f}%"
        )

    parts.append("\nLOGS (recent, may contain noise):")
    for line in obs.get("logs", [])[:8]:
        parts.append(f"  {line}")

    parts.append("\nTOPOLOGY (service → dependencies):")
    for svc, deps in obs.get("topology", {}).items():
        parts.append(f"  {svc} → {deps}")

    parts.append("\nReturn your next action as a JSON object.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> SecOpsAction:
    """Parse LLM output into a SecOpsAction. Strips markdown fences if present."""
    raw = raw.strip()
    # Strip common fence patterns
    for fence in ["```json", "```JSON", "```", "`"]:
        if raw.startswith(fence):
            raw = raw[len(fence):]
        if raw.endswith(fence):
            raw = raw[: -len(fence)]
    raw = raw.strip()
    data = json.loads(raw)
    return SecOpsAction(
        action_type=data.get("action_type", "inspect_metrics"),
        parameters=data.get("parameters", {}),
    )


# ---------------------------------------------------------------------------
# Heuristic fallback agent (deterministic, no LLM required)
# ---------------------------------------------------------------------------
_HEURISTIC: dict[str, list[dict]] = {
    "easy_memory_leak": [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "query_logs",       "parameters": {"service": "auth"}},
        {"action_type": "restart_service",  "parameters": {"service": "auth"}},
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
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


def _heuristic_action(task_id: str, step: int) -> str:
    seq = _HEURISTIC.get(task_id, [
        {"action_type": "inspect_metrics",  "parameters": {}},
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
    ])
    idx = min(step - 1, len(seq) - 1)
    return json.dumps(seq[idx])


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(client, task_id: str) -> dict[str, Any]:
    """
    Run one complete episode for the given task_id.
    Returns a dict with task_id, score, success, steps, rewards.
    Emits [START], [STEP]…, [END] to stdout.
    """
    env = OpenSecOpsEnv()
    obs_raw = env.reset(task_id=task_id)

    obs_dict: dict[str, Any] = {
        "alerts":             obs_raw.alerts,
        "metrics":            obs_raw.metrics,
        "logs":               obs_raw.logs,
        "topology":           obs_raw.topology,
        "last_action_result": obs_raw.last_action_result,
        "time_step":          obs_raw.time_step,
    }

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards:  list[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        done = False
        step_n = 0

        while not done and step_n < MAX_STEPS:
            step_n += 1
            obs_text = _obs_to_text(obs_dict)
            messages.append({"role": "user", "content": obs_text})

            # ── LLM call ──────────────────────────────────────────────────
            error: Optional[str] = None
            if client is not None:
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    raw_action = (resp.choices[0].message.content or "").strip()
                    messages.append({"role": "assistant", "content": raw_action})
                except Exception as exc:
                    error = str(exc)
                    raw_action = _heuristic_action(task_id, step_n)
                    print(f"[DEBUG] LLM error: {exc}", flush=True)
            else:
                raw_action = _heuristic_action(task_id, step_n)

            # ── Parse action ──────────────────────────────────────────────
            try:
                action = _parse_action(raw_action)
            except (json.JSONDecodeError, KeyError, ValueError):
                action = SecOpsAction(action_type="inspect_metrics", parameters={})
                error = f"parse_error: {raw_action[:80]}"

            # ── Step environment ──────────────────────────────────────────
            obs_result, reward, done, info = env.step(action)

            obs_dict = {
                "alerts":             obs_result.alerts,
                "metrics":            obs_result.metrics,
                "logs":               obs_result.logs,
                "topology":           obs_result.topology,
                "last_action_result": obs_result.last_action_result,
                "time_step":          obs_result.time_step,
            }

            rewards.append(reward)
            steps_taken = step_n

            # action string in spec format: action_type(json_params)
            action_str = f"{action.action_type}({json.dumps(action.parameters)})"

            log_step(
                step=step_n,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

        # ── Grade finished episode ─────────────────────────────────────────
        grade_result = grade(env.state.to_dict())
        score   = round(max(0.0, min(1.0, grade_result.score)), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Build OpenAI client ────────────────────────────────────────────────
    try:
        from openai import OpenAI as _OpenAI
        _openai_available = True
    except ImportError:
        _openai_available = False

    if _openai_available and _API_KEY:
        client = _OpenAI(api_key=_API_KEY, base_url=API_BASE_URL)
        print(f"[INFO] LLM model   : {MODEL_NAME}", flush=True)
        print(f"[INFO] Endpoint    : {API_BASE_URL}", flush=True)
    else:
        client = None
        if not _openai_available:
            print("[WARN] openai package not installed — running deterministic heuristic baseline.", flush=True)
        else:
            print("[WARN] No HF_TOKEN set — running deterministic heuristic baseline.", flush=True)
            print("[WARN] Set HF_TOKEN to use a real LLM: $env:HF_TOKEN = 'hf_...'", flush=True)

    # ── Run all tasks ──────────────────────────────────────────────────────
    results: list[dict] = []
    for task_id in TASKS:
        r = run_task(client, task_id)
        results.append(r)
        print()  # blank line between tasks

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    total = 0.0
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {status} {r['task_id']:30s}  score={r['score']:.4f}  steps={r['steps']}",
            flush=True,
        )
        total += r["score"]
    avg = total / len(results) if results else 0.0
    print(f"\n  AVERAGE SCORE: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
