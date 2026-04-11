---
title: OpenSecOpsEnv
emoji: 🔐
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - secops
  - incident-response
  - agent-evaluation
---

# 🔐 OpenSecOpsEnv

> **An OpenEnv-compliant incident response environment where an AI agent acts as an on-call security engineer.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Motivation

Modern cloud operations sit at the intersection of **DevOps** and **SecOps**.
A single production incident may be caused by:

- An infrastructure failure (memory leak, OOM killer, service crash)
- A misconfiguration after a bad deployment
- An active cyber attack (DDoS, data exfiltration, privilege escalation)

…and the signals for all three often *look the same* at first glance.

**OpenSecOpsEnv** provides a realistic, reproducible benchmark where agents must:

1. **Investigate** a running distributed system (query logs, inspect metrics, run scans)
2. **Correlate** noisy, partial, sometimes misleading signals
3. **Mitigate** the specific root cause (block IPs, restart services, isolate compromised nodes)
4. **Diagnose** the incident with a precise label

This directly addresses the *SecOps skills gap*: can LLMs / RL agents serve as effective first-responders?

---

## 📦 Project Structure

```
├── inference.py                 # Baseline inference script (root, as required)
├── openenv.yaml                 # OpenEnv manifest
├── Dockerfile                   # Production container
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── server/
│   ├── __init__.py
│   └── app.py                   # Root-level entry point (for openenv validate)
├── opensecops_env/
│   ├── __init__.py              # Package exports
│   ├── models.py                # SecOpsAction, SecOpsObservation, SecOpsState (dataclasses)
│   ├── env.py                   # OpenSecOpsEnv – core environment (reset/step/state)
│   ├── grader.py                # Deterministic multi-component grader → [0, 1]
│   ├── client.py                # HTTP client for remote server interaction
│   └── tasks/
│       ├── __init__.py
│       └── task_definitions.py  # 4 task configs (easy → medium → medium-hard → hard)
│   └── server/
│       ├── __init__.py
│       └── app.py               # FastAPI server (reset/step/state/grade endpoints)
└── tests/
    └── test_opensecops.py       # 33 unit tests
```

---

## 🔍 Observation Space

Each step the agent receives a `SecOpsObservation` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `alerts` | `list[dict]` | Threshold-based monitoring alerts `{service, type, severity, message}` |
| `metrics` | `dict[str, dict]` | Per-service `{cpu, memory, latency, error_rate}` snapshots |
| `logs` | `list[str]` | Recent (partial, noisy) log lines — up to 8 lines per step |
| `topology` | `dict[str, list]` | Service dependency graph `{svc → [deps]}` |
| `last_action_result` | `str` | Human-readable result of the previous action |
| `time_step` | `int` | Current step index |
| `available_actions` | `list[str]` | All valid action types |

> ⚠️ **Observations are intentionally partial.** The true root cause is never directly exposed. The agent must infer it.

---

## 🎮 Action Space

All actions use a `SecOpsAction(action_type, parameters)` structure:

| `action_type` | Parameters | Description |
|--------------|-----------|-------------|
| `query_logs` | `{"service": "<name>"}` | Fetch recent log lines for a service |
| `inspect_metrics` | `{"service": "<name>"}` or `{}` | View metric snapshot |
| `restart_service` | `{"service": "<name>"}` | Restart a service (resets its metrics) |
| `scale_service` | `{"service": "<name>", "replicas": <n>}` | Scale out to handle load |
| `block_ip` | `{"ip": "<addr>"}` | Block an IP at the network boundary |
| `rollback_deployment` | `{"service": "<name>", "version": "<tag>"}` | Rollback to previous version |
| `run_security_scan` | `{"target": "<name>"}` | Run deep security scan on a service |
| `isolate_service` | `{"service": "<name>"}` | Network-isolate a compromised service |
| `submit_diagnosis` | `{"label": "<root_cause>:<subtype>"}` | **Terminal action** – finalise diagnosis |

### Diagnosis Labels

```
infra_failure:memory_leak
infra_failure:service_crash
misconfiguration:bad_config
cyber_attack:ddos
cyber_attack:data_exfiltration
cyber_attack:privilege_escalation
```

---

## 📋 Tasks (4 tasks, easy → hard)

### Task 1 – EASY: `easy_memory_leak` (seed=42)
**Scenario:** The `auth` service has a progressive memory leak.  
**Signals:** Clear log messages, steadily rising memory metrics, single service affected.  
**Correct actions:** Inspect metrics → Query auth logs → Restart auth → Submit `infra_failure:memory_leak`  
**Max steps:** 30 | **Noise:** 5%

### Task 2 – MEDIUM: `medium_ddos_cascade` (seed=1337)
**Scenario:** DDoS attack from two IP ranges cascades through gateway → api → auth.  
**Signals:** Multiple services degraded, requires IP correlation from logs across api and auth.  
**Correct actions:** Block both IPs (203.0.113.45, 198.51.100.12) → Scale api → Submit `cyber_attack:ddos`  
**Max steps:** 40 | **Noise:** 25%

### Task 3 – MEDIUM-HARD: `medium_hard_bad_deployment` (seed=9999)
**Scenario:** A bad `api` v2.4.1 deployment pushed an invalid Redis connection string, sending the cache service into a reconnect storm. False gateway alerts distract from the real cause.  
**Signals:** Deployment timestamp in logs correlates with degradation onset. Cache error logs show connection refused.  
**Correct actions:** Inspect metrics → Query api/cache logs → Rollback api deployment → Restart cache → Submit `misconfiguration:bad_config`  
**Max steps:** 45 | **Noise:** 35%

### Task 4 – HARD: `hard_data_exfiltration` (seed=31337)
**Scenario:** Compromised service account (`reports_bot`) exfiltrating 4+ GB of data from the DB to an external host. False alerts on cache service to mislead.  
**Signals:** Buried in noisy logs, 55% noise, false critical alert on cache.  
**Correct actions:** Run security scans on db+auth → Isolate db → Block 10.0.0.99 → Submit `cyber_attack:data_exfiltration`  
**Max steps:** 50 | **Noise:** 55%

---

## 🏆 Reward Function

Dense rewards are provided at every step (with **diminishing returns** for repeated investigation):

| Event | Reward |
|-------|--------|
| Useful investigation (affected service) | **+0.2** |
| Correct intermediate inference (security scan hits) | **+0.3** |
| Correct mitigation step | **+0.5** |
| Correct final diagnosis | **+1.0** |
| Irrelevant investigation | **-0.05** |
| Ineffective mitigation | **-0.1** |
| Harmful action (blocking legit IP / isolating healthy service) | **-0.5** |
| Wrong diagnosis | **-1.0** |

---

## 📊 Grader

Each episode is scored in **[0, 1]**:

```
score = 0.5 × diagnosis_correct
      + 0.3 × action_efficiency
      + 0.2 × investigation_quality
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `diagnosis_correct` | 50% | 1.0 exact match, 0.5 correct category, 0.0 wrong |
| `action_efficiency` | 30% | Fraction of correct mitigations achieved, adjusted for steps used |
| `investigation_quality` | 20% | Fraction of affected services investigated |

### Baseline Scores (Heuristic Agent)

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| `easy_memory_leak` | Easy | **1.000** | 4 |
| `medium_ddos_cascade` | Medium | **1.000** | 8 |
| `medium_hard_bad_deployment` | Medium-Hard | **1.000** | 6 |
| `hard_data_exfiltration` | Hard | **1.000** | 8 |
| **Average** | | **1.000** | |

> These are deterministic heuristic scores. A frontier 72B LLM typically scores ~0.75–0.85 on the harder tasks, showing the benchmark discriminates agent capability.

---

## 🚀 Setup

### Local (no Docker)

```bash
# Clone & enter project
cd opensecops_env

# Install
pip install -e ".[dev]"

# Run tests (33 tests)
pytest tests/ -v

# Start server
uvicorn opensecops_env.server.app:app --host 0.0.0.0 --port 8000

# Run baseline inference
python inference.py
```

### Docker

```bash
# Build
docker build -t opensecops-env:latest .

# Run
docker run -p 8000:8000 opensecops-env:latest
```

### Inference with LLM

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

python inference.py
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes (for LLM) | *none* | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint URL |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `LOCAL_IMAGE_NAME` | No | *none* | Docker image for `from_docker_image()` |

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe → `{"status": "ok"}` |
| `GET` | `/tasks` | List available tasks with metadata |
| `POST` | `/reset` | Start episode: `{"task_id": "easy_memory_leak"}` |
| `POST` | `/step` | Execute action: `{"action_type": "...", "parameters": {...}}` |
| `GET` | `/state` | Full internal state (for debugging / graders) |
| `POST` | `/grade` | Grade current episode → `{score, details}` |
| `GET` | `/web` | Interactive debug UI (set `ENABLE_WEB_INTERFACE=true`) |

---

## 🧪 Validation

```bash
# Local spec validation
pip install openenv-core
openenv validate

# Run unit tests
pytest tests/ -v    # 33 tests
```

---

## 🧠 Design Principles

1. **Multi-step reasoning required** — no single action resolves any task
2. **Partial observability** — root cause never directly visible
3. **Noisy signals** — misleading logs and false alerts on all tasks  
4. **Action consequences** — blocking wrong IP harms users; isolating healthy service causes outage
5. **Deterministic reproducibility** — fixed seeds ensure identical episodes
6. **Dense reward** — non-zero signal at every step guides learning
7. **Diminishing returns** — repeated investigation of the same service yields less reward
8. **4 difficulty levels** — easy → medium → medium-hard → hard progression

---

## 📜 License

MIT License — see [LICENSE](LICENSE).
