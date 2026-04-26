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
  - multi-agent
  - grpo
  - curriculum-learning
---

# 🔐 OpenSecOpsEnv — AI Security Engineer That Actually Learns

> **A multi-agent OpenEnv environment where an AI Defender battles a live Attacker to resolve real production security incidents — and gets smarter with every episode.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv/openenv)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-orange)](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training)
[![Model](https://img.shields.io/badge/🤗%20Trained%20Model-Qwen2.5--7B--GRPO-green)](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎬 The Problem

Every hour a security incident goes unresolved costs thousands of dollars and puts user data at risk. Yet the on-call engineer gets paged at 3AM staring at a wall of noisy, contradictory alerts:

- Is that CPU spike a **memory leak** or a **DDoS attack**?
- Is that suspicious IP a **real attacker** or a **false alert planted by the attacker**?
- Should you **restart the service** or **isolate it**? One wrong action makes things worse.

**Can an LLM learn to be that expert, battle-hardened on-call engineer?**

OpenSecOpsEnv is a realistic, reproducible benchmark for answering exactly that question.

---

## 🏗️ Environment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenSecOpsEnv                                │
│                                                                   │
│   Production Incident                                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│   │ gateway  │──▶│   api    │──▶│  cache   │──▶│    db    │    │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│         │               │              │               │         │
│         └───────────────┴──────────────┴───────────────┘        │
│                              │                                    │
│                    ┌─────────▼─────────┐                         │
│                    │  auth service     │                          │
│                    └───────────────────┘                          │
│                                                                   │
│   🔴 Red Agent (Attacker)    🔵 Blue Agent (Defender / LLM)      │
│   - Injects noise            - Queries logs                       │
│   - Amplifies attacks        - Inspects metrics                   │
│   - Creates false alerts     - Runs security scans               │
│   - Spreads to new services  - Blocks IPs, isolates services     │
│                              - Submits diagnosis                  │
└─────────────────────────────────────────────────────────────────┘
```

The environment is **fully adversarial**: while the Blue Agent (your trained LLM) investigates and mitigates, a heuristic Red Agent is actively escalating the attack, injecting noise, and creating false alerts to slow it down.

---

## 🤖 What Makes This Unique

### 1. Multi-Agent Adversarial Battle
Unlike typical benchmark environments, OpenSecOpsEnv features a live **Red (Attacker) vs Blue (Defender)** dynamic. The Red Agent:
- Injects misleading log entries to obscure the root cause
- Amplifies attack progress in real time
- Corrupts healthy service metrics to create false alarms
- Spreads the attack to adjacent services in the topology

This makes the Defender's task genuinely hard and tests **theory-of-mind reasoning**: the agent must distinguish between real signals and adversarially planted noise.

### 2. Curriculum Self-Improvement
The Blue Agent starts at Level 1 (easy memory leaks) and **automatically levels up** when it achieves a rolling average score above the threshold. The 5-level curriculum goes:

```
Level 1: Easy memory leaks (threshold: 0.65)
Level 2: + Medium DDoS cascade (threshold: 0.70)
Level 3: + Bad deployment scenarios (threshold: 0.72)
Level 4: + Hard data exfiltration (threshold: 0.75)
Level 5: Hard exfiltration only (threshold: 0.80)
```

### 3. Partial Observability + Adversarial Noise
The true root cause is **never directly observable**. The agent sees:
- Noisy, partial log lines (up to 8 per step)
- Metric snapshots that may be artificially spiked by the Red Agent
- False critical alerts on healthy services
- Up to **55% noise ratio** on the hardest task

---

## 📊 Training Results

We fine-tuned **Qwen2.5-7B-Instruct** using **GRPO (Group Relative Policy Optimization)** for 500 steps on the OpenSecOpsEnv reward signal.

![Training Results](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo/resolve/main/training_results.png)

### Before vs After (Episode Score [0, 1])

| Task | Difficulty | Untrained | GRPO-Trained | Improvement |
|------|-----------|-----------|--------------|-------------|
| Memory Leak | Easy | 0.51 | **0.95** | +86% |
| DDoS Cascade | Medium | 0.38 | **0.87** | +129% |
| Bad Deployment | Medium-Hard | 0.31 | **0.81** | +161% |
| Data Exfiltration | Hard | 0.22 | **0.76** | +245% |

> The hardest task (data exfiltration, 55% noise, active Red Agent spreading the attack) shows the most dramatic improvement — from near-random (0.22) to reliable expert-level (0.76).

---

## 🎮 Tasks — 4 Difficulty Levels

### Task 1 — EASY: `easy_memory_leak`
**Scenario:** The `auth` service has a progressive memory leak.
**Key challenge:** Distinguish memory leak from fake CPU alerts injected by the Red Agent.
**Correct diagnosis:** `infra_failure:memory_leak`

### Task 2 — MEDIUM: `medium_ddos_cascade`
**Scenario:** DDoS attack from two IPs cascades through gateway → api → auth.
**Key challenge:** Correlate IP addresses buried in logs across 3 services.
**Correct diagnosis:** `cyber_attack:ddos`

### Task 3 — MEDIUM-HARD: `medium_hard_bad_deployment`
**Scenario:** Bad `api` v2.4.1 deployment breaks Redis connections. False gateway alerts distract.
**Key challenge:** Correlate deployment timestamp with degradation onset across services.
**Correct diagnosis:** `misconfiguration:bad_config`

### Task 4 — HARD: `hard_data_exfiltration`
**Scenario:** Compromised service account exfiltrating 4+ GB. Red Agent actively spreads attack.
**Key challenge:** Find real signal in 55% noise + false critical alert planted on cache.
**Correct diagnosis:** `cyber_attack:data_exfiltration`

---

## 🏆 Reward Function

Dense rewards at **every step** (not binary — can't be gamed):

| Event | Reward |
|-------|--------|
| Useful investigation (affected service) | **+0.20** |
| Correct security scan on affected service | **+0.30** |
| Correct mitigation step | **+0.50** |
| Correct final diagnosis | **+1.00** |
| Irrelevant investigation | **-0.05** |
| Ineffective mitigation | **-0.10** |
| Harmful action (blocking legit IP/isolating healthy service) | **-0.50** |
| Wrong diagnosis | **-1.00** |
| Step cost | **-0.02** |

### Episode Grader
```
score = 0.5 × diagnosis_correct
      + 0.3 × action_efficiency
      + 0.2 × investigation_quality
```

---

## 🚀 Live Demo

▶️ **[Launch Dashboard](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training)**

The live dashboard features three modes:
1. **Agent Demo** — Watch Trained vs Untrained side-by-side on any of the 4 tasks
2. **Battle Mode** — Live Red Attacker vs Blue Defender stream with real-time reward tracking
3. **Self-Improvement** — Curriculum level tracker showing the agent levelling up

---

## 🧪 Training Notebook

▶️ **[Open in JupyterLab on HF](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training)** (Run the colab_training.ipynb)

The notebook:
- Loads Qwen2.5-7B-Instruct with 4-bit quantization via Unsloth
- Applies LoRA adapters (r=16, target all attention + MLP projections)
- Trains using `trl.GRPOTrainer` with our custom `secops_reward_fn`
- Logs reward + loss curves
- Pushes the merged 16-bit model to HF Hub

```python
# Core reward function — wraps the environment directly
def secops_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        action = parse_action(completion)
        if action is None:
            rewards.append(-0.5)   # JSON format penalty
            continue
        env = OpenSecOpsEnv()
        env.reset(task_id)
        _, reward, _, _ = env.step(action)
        rewards.append(float(reward) - 0.02)  # step cost
    return rewards
```

---

## 📦 Project Structure

```
├── colab_training.ipynb         # 🔑 Full GRPO training notebook
├── inference.py                 # Baseline inference (OpenEnv required)
├── openenv.yaml                 # OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── opensecops_env/
│   ├── env.py                   # Core environment (reset/step/state)
│   ├── grader.py                # Multi-component grader → [0, 1]
│   ├── models.py                # SecOpsAction, Observation, State
│   ├── tasks/task_definitions.py # 4 task configs
│   └── server/app.py            # FastAPI + live battle SSE streams
└── tests/test_opensecops.py     # 33 unit tests
```

---

## 🏃 Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests (33 tests)
pytest tests/ -v

# Start server
uvicorn opensecops_env.server.app:app --host 0.0.0.0 --port 8000

# Open dashboard
open http://localhost:8000/dashboard
```

### Docker
```bash
docker build -t opensecops-env:latest .
docker run -p 8000:8000 opensecops-env:latest
```

---

## 🔗 All Links

| Resource | Link |
|----------|------|
| 🤗 HF Space (Live Demo) | https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training |
| 🧠 Trained Model | https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo |
| 📓 Training Notebook | [colab_training.ipynb](./colab_training.ipynb) |
| 🎥 Demo Video | _Coming soon_ |
| 📝 Blog Post | _Coming soon_ |

---

## 🧠 Design Principles

1. **Multi-step reasoning required** — no single action resolves any task
2. **Partial observability** — root cause never directly visible
3. **Adversarially noisy** — misleading logs and Red-Agent-planted false alerts
4. **Action consequences** — wrong actions actively harm the system (negative rewards)
5. **Deterministic reproducibility** — fixed seeds for fair comparison
6. **Dense reward** — non-zero signal at every step guides RL training
7. **Curriculum progression** — 5 levels of difficulty for self-improvement

---

## 📜 License

MIT License — see [LICENSE](LICENSE).
