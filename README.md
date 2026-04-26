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

# 🔐 OpenSecOpsEnv — AI Security Engineer That Learns to Beat Cyberattacks

> **A multi-agent adversarial OpenEnv environment where a GRPO-trained Qwen2.5-7B must resolve real production security incidents — while a live Attacker agent actively tries to sabotage the investigation with injected false alerts and corrupted metrics.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv/openenv)
[![HF Space](https://img.shields.io/badge/🤗_HF_Space-Live_Demo-orange)](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training)
[![Model](https://img.shields.io/badge/🤗_Trained_Model-Qwen2.5--7B--GRPO-green)](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎬 The Problem

Every hour a security incident goes unresolved costs thousands of dollars and puts user data at risk. The on-call engineer gets paged at 3AM and stares at a wall of contradictory alerts:

```
[CRITICAL] cache  · memory 82%          ← FAKE — planted by the attacker
[WARNING]  db     · CPU 82%             ← real noise
[CRITICAL] auth   · memory 89%          ← real signal
[WARNING]  db     · Outbound to 10.0.0.99 ← THE REAL ATTACK
```

A junior engineer panics and restarts the cache service. The real exfiltration continues for 40 more minutes. **4GB of customer data is gone.**

**Can a language model learn to not make that mistake — even under active adversarial sabotage?**

---

## 📊 Results — GRPO Training on OpenSecOpsEnv

We fine-tuned **Qwen2.5-7B-Instruct** using **GRPO** for 500 steps on our environment's reward signal.

![Training Results — Reward curve, GRPO loss, and Before vs After scores](./training_results.png)
*Left: Per-step reward rises from ~0.10 (random) to stable ~0.20 (trained). Centre: GRPO policy loss rising = gradient actively learning reward differences between candidate actions. Right: Episode scores improve 86–245% across all 4 tasks.*

| Task | Difficulty | Untrained | GRPO-Trained | Improvement |
|------|-----------|-----------|--------------|-------------|
| Memory Leak | Easy | 0.51 | **0.95** | +86% |
| DDoS Cascade | Medium | 0.35 | **0.87** | +149% |
| Bad Deployment | Medium-Hard | 0.31 | **0.81** | +161% |
| **Data Exfiltration** | **Hard** | **0.22** | **0.76** | **+245%** |

> **The hardest task shows the most dramatic improvement.** The untrained model scores 0.22 — essentially random. After GRPO, it reaches 0.76: reliably ignoring the attacker's planted false alert, identifying the real exfiltration, and submitting the correct diagnosis.

**Why the reward curve plateaus at ~0.20/step:** Most steps are neutral investigations (+0.0 reward). Terminal rewards (+1.0 correct diagnosis, -1.0 wrong) only fire once per episode. Episode-level score (the table above) is the right metric — and it shows 2–4× improvement across every difficulty level.

**Why the loss curve rises:** In GRPO, rising policy loss means the gradient is active — the model is differentiating between candidate action sequences and updating weights based on environment feedback. This is the expected healthy training signature.

---

## 🏗️ Environment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenSecOpsEnv                                │
│                                                                   │
│  Production Microservices (Partially Observable)                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ gateway  │──▶│   api    │──▶│  cache   │──▶│    db    │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│        │               └──────────────────────────▶ auth        │
│                                                                   │
│  🔴 Red Agent (Attacker) — adaptive adversarial (theory-of-mind)  │
│     · counter_investigate: Blue queries db → Red plants alert on auth│
│     · amplify_attack    : boosts attack before Blue can contain it  │
│     · accelerate_spread : spreads to services Blue hasn't checked   │
│     · corrupt_metric    : spikes a service Blue already looked at   │
│     · inject_noise      : misleading log entries (default)          │
│                                                                   │
│  🔵 Blue Agent (Defender) — your trained Qwen2.5-7B-GRPO       │
│     · query_logs         : read service logs                     │
│     · inspect_metrics    : view all service metrics              │
│     · run_security_scan  : deep scan a service                   │
│     · restart_service    : restart a crashing service            │
│     · scale_service      : scale replicas                        │
│     · block_ip           : block an attacking IP                 │
│     · rollback_deployment: revert a bad deployment               │
│     · isolate_service    : cut a service from the network        │
│     · submit_diagnosis   : final answer (root cause label)       │
└─────────────────────────────────────────────────────────────────┘
```

**Core design principles:**
1. **Partial observability** — root cause and attack progress are **never directly visible**
2. **Dense rewards** — non-zero signal at every step (not binary success/fail)
3. **Action consequences** — wrong isolations/restarts harm the system (negative rewards)
4. **Adversarial noise** — Red Agent corrupts observations in real time
5. **Random seeds** — different starting states every episode; no memorization possible
6. **Reproducible grading** — weighted composite score across 3 components

---

## 🤖 What Makes This Novel

### 1. True Multi-Agent Adversarial Competition
The Red Agent and Blue Agent share the **same live environment state**. Red's `create_false_alert` action literally appends a new alert to the observation the Blue Agent will see next step. This is not simulated interference — it's real shared state mutation. The Blue Agent must reason about WHY an alert might be adversarially planted.

### 2. Curriculum Self-Improvement (5 Levels)
The Blue Agent starts at Level 1 and **automatically earns harder scenarios** as it improves:

```
Level 1: easy_memory_leak              → avg score ≥ 0.65 over 5 episodes
Level 2: + medium_ddos_cascade         → avg score ≥ 0.70
Level 3: + medium_hard_bad_deployment  → avg score ≥ 0.72
Level 4: + hard_data_exfiltration      → avg score ≥ 0.75
Level 5: hard_data_exfiltration only   → expert-level only
```

### 3. Designed to Foil Shortcuts
- **You can't guess the answer** — wrong diagnosis gives -1.0, ends episode with ~0 score
- **You can't spam safe actions** — every step costs -0.02, step limit enforced
- **You can't ignore the Red Agent** — false alerts actively mislead if the defender doesn't reason carefully
- **Each run is different** — random seed jitter on all starting metrics prevents memorization

---

## 🏆 Reward Function

Dense rewards at **every step** (can't be gamed with binary success):

| Event | Reward | Why |
|-------|--------|-----|
| Investigate affected service (logs/metrics) | **+0.20** | Reward targeted investigation |
| Security scan on affected service | **+0.30** | Reward hypothesis-driven scanning |
| Correct mitigation (right service/IP) | **+0.50** | Reward precise action |
| Correct final diagnosis | **+1.00** | Maximum reward |
| Irrelevant investigation | **-0.05** | Discourage scatter-gun approach |
| Ineffective mitigation | **-0.10** | Penalise wasted actions |
| Harmful action (wrong isolate/IP) | **-0.50** | Hard penalty: making things worse |
| Wrong final diagnosis | **-1.00** | Maximum penalty |
| Per-step cost | **-0.02** | Efficiency pressure |

### Episode Grader
```
score = 0.5 × diagnosis_correct
      + 0.3 × action_efficiency
      + 0.2 × investigation_quality
```

---

## 🎯 4 Tasks — Increasing Difficulty

### Task 1 — EASY: Memory Leak in auth
- **What's happening:** `auth` service has a progressive memory leak
- **What you see:** Memory rising, latency climbing
- **Correct response:** `query_logs(auth)` → `restart_service(auth)` → `submit_diagnosis(infra_failure:memory_leak)`
- **Noise:** 5% — clear signals

### Task 2 — MEDIUM: DDoS Cascade Attack  
- **What's happening:** Two external IPs flooding `gateway` → cascading to `api` → `auth`
- **What you see:** High error rates across 3 services simultaneously
- **Correct response:** Review gateway logs, block both IPs, scale `api`
- **Noise:** 25% — ambiguous multi-service signals

### Task 3 — MEDIUM-HARD: Bad Deployment
- **What's happening:** `api` v2.4.1 pushed invalid Redis config → reconnect storm
- **What you see:** `cache` high CPU (gateway alert misdirects you here)
- **Correct response:** Identify deployment timestamp correlation, rollback `api`
- **Noise:** 35% — false gateway alert distracts

### Task 4 — HARD: Data Exfiltration (Disguised)
- **What's happening:** Compromised `reports_bot` service account exfiltrating 4GB+ via `db`
- **What you see:** Fake CRITICAL cache alert (planted by Red Agent) + subtle db outbound traffic
- **Correct response:** Ignore cache, scan `db`, identify `reports_bot`, isolate `db`, block `10.0.0.99`
- **Noise:** **55%** — plus active Red Agent spreading attack and amplifying it

---

## 🚀 Live Demo Dashboard

▶️ **[Launch Dashboard](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training)**

Three modes:

**Agent Tab:** Watch your trained Qwen2.5-7B-GRPO investigate and resolve incidents in real time. Toggle between Trained vs Baseline (untrained) to see the contrast. Every action card shows the raw JSON the model generated — verifiable live inference, not a replay.

**Battle Tab:** Live Red vs Blue adversarial stream. Watch the attacker inject fake alerts mid-episode. Watch the trained model resist the mislead and correctly identify the real attack vector. Score tracked per-round with Defender Advantage and Attack Suppression metrics.

**Learning Tab:** Curriculum progress across sessions. As you run episodes, scores are tracked server-side, rolling averages computed, and level-up events recorded.

---

## 🧪 Training — Reproducible Notebook

▶️ **[colab_training.ipynb](./colab_training.ipynb)** — Full GRPO training on HF Spaces (A100)

```python
# Core reward function — wraps the environment directly
def secops_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        action = parse_action(completion)        # Parse JSON from LLM output
        if action is None:
            rewards.append(-0.5)                # JSON format penalty
            continue
        env = OpenSecOpsEnv()
        env.reset(task_id)
        _, reward, _, _ = env.step(action)       # Execute in environment
        rewards.append(float(reward) - 0.02)    # Apply step cost
    return rewards

trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(num_generations=4, max_new_tokens=128, temperature=0.9),
    reward_funcs=secops_reward_fn,
    train_dataset=dataset,
)
trainer.train()
```

**Setup:** Qwen2.5-7B-Instruct + Unsloth 4-bit + LoRA (r=16) + TRL GRPOTrainer. Merged to 16-bit for clean production deployment. Tracked with W&B.

---

## 📦 Project Structure

```
├── colab_training.ipynb              # ← Full GRPO training (run this)
├── opensecops_env/
│   ├── env.py                        # Core OpenEnv environment (reset/step/state)
│   ├── grader.py                     # Multi-component grader [0,1]
│   ├── models.py                     # SecOpsAction, Observation, HiddenState
│   ├── tasks/task_definitions.py     # 4 incident configs with full metadata
│   └── server/app.py                 # FastAPI + SSE streams + live dashboard
├── tests/test_opensecops.py          # 33 unit tests (all passing)
├── hf_blog_post.md                   # Full technical writeup
├── DASHBOARD_GUIDE.md                # Plain-English dashboard explanation
├── TECHNICAL_ANALYSIS.md             # Full pipeline + theme alignment analysis
├── openenv.yaml                      # OpenEnv manifest
├── Dockerfile
└── requirements.txt
```

---

## 🏃 Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run test suite (33 tests)
pytest tests/ -v

# Start server (auto-loads .env for HF_TOKEN)
uvicorn opensecops_env.server.app:app --host 0.0.0.0 --port 8000

# Open dashboard
open http://localhost:8000/dashboard

# Test live AI endpoint
open http://localhost:8000/debug/ai
```

**Environment variables (.env file):**
```
HF_TOKEN=hf_xxxx                         # Required for live AI inference
TRAINED_MODEL_ENDPOINT=https://...       # Override default endpoint
```

### Docker
```bash
docker build -t opensecops-env:latest .
docker run -p 8000:8000 -e HF_TOKEN=hf_xxxx opensecops-env:latest
```

---

## 🔗 All Resources

| Resource | Link |
|----------|------|
| 🤗 HF Space (Live Demo + Training) | [SapphireGaze429/opensecops-grpo-training](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training) |
| 🧠 Trained Model | [SapphireGaze429/opensecops-qwen2.5-7b-grpo](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo) |
| 📓 Training Notebook | [colab_training.ipynb](./colab_training.ipynb) |
| 📊 Training Plots | [training_results.png](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo/resolve/main/training_results.png) |
| 📝 Full Blog Post | [hf_blog_post.md](./hf_blog_post.md) |
| 📖 Dashboard Guide | [DASHBOARD_GUIDE.md](./DASHBOARD_GUIDE.md) |
| 🔬 Technical Analysis | [TECHNICAL_ANALYSIS.md](./TECHNICAL_ANALYSIS.md) |

---

## 🏆 Scoring Criteria Alignment

| Criterion | Weight | How We Address It |
|-----------|--------|------------------|
| **Environment Innovation** | 40% | Multi-agent adversarial with shared live state; partial observability; 55% adversarial noise; curriculum self-improvement; dense reward with action consequences |
| **Storytelling** | 30% | Live dashboard with real AI output; before/after demo; Battle Mode with visible attacker vs defender; Dashboard Guide + Blog Post |
| **Showing Improvement** | 20% | +245% on hardest task; reward and loss curves; before/after bar chart across all 4 difficulty levels |
| **Reward + Training Pipeline** | 10% | Dense multi-component reward; GRPO with environment-derived signal; reproducible notebook; W&B tracking |

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

*Built for the OpenEnv Hackathon Round 2. Every dashboard action is live model inference against the HF Inference Endpoint — no mock data, no pre-scripted replays.*
