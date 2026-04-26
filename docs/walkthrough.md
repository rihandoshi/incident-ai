# OpenSecOpsEnv — Full Project Walkthrough

## What Is This?

**OpenSecOpsEnv** is an OpenEnv-compliant reinforcement learning environment where an AI agent acts as an on-call security engineer. The agent must investigate and resolve production incidents across 4 escalating scenarios — from a simple memory leak to a disguised data exfiltration attack buried under 55% noise.

---

## 🤖 What Your Agent Does — Full Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                     Episode Flow                               │
│                                                                │
│  env.reset(task_id)                                            │
│       │                                                        │
│       ▼                                                        │
│  HiddenState (true root cause — NEVER shown to agent)         │
│       │ drives                                                 │
│       ▼                                                        │
│  SecOpsObservation (alerts + metrics + logs + topology)        │
│       │                                                        │
│       ▼                                                        │
│  LLM reads observation as text prompt                          │
│       │                                                        │
│       ▼                                                        │
│  LLM outputs JSON: {"action_type": "...", "parameters": {...}} │
│       │                                                        │
│       ▼                                                        │
│  env.step(action) → (new_obs, reward, done, info)             │
│       │                                                        │
│  GRPO uses reward to update model weights                      │
│       │                                                        │
│       └──── loop until done or max_steps ────────────────────┘
│                                                                │
│  grade(state) → score [0, 1]                                  │
└────────────────────────────────────────────────────────────────┘
```

### The 4 Tasks

| Task | Incident | Root Cause | Noise | Max Steps |
|------|----------|------------|-------|-----------|
| easy_memory_leak | Auth service heap overflow | `infra_failure:memory_leak` | 5% | 30 |
| medium_ddos_cascade | DDoS from 2 IP ranges → gateway → api → auth | `cyber_attack:ddos` | 25% | 40 |
| medium_hard_bad_deployment | Bad Redis config in api v2.4.1 → cache reconnect storm | `misconfiguration:bad_config` | 35% | 45 |
| hard_data_exfiltration | Compromised `reports_bot` exfiltrating 4GB via DB, false cache alert | `cyber_attack:data_exfiltration` | 55% | 50 |

### Reward Design (Dense)

| Action | Reward |
|--------|--------|
| Investigate affected service (first time) | +0.20 |
| Security scan hits cyber attack target | +0.30 |
| Correct mitigation step | +0.50 |
| Correct diagnosis | +1.00 |
| Investigate irrelevant service | −0.05 |
| Wrong mitigation (e.g., restart wrong service) | −0.10 |
| Harmful action (block legit IP, isolate healthy service) | −0.50 |
| Wrong diagnosis | −1.00 |

### Grader (Episode Score [0, 1])

```
score = 0.5 × diagnosis_correct
      + 0.3 × action_efficiency
      + 0.2 × investigation_quality
```

---

## 🧠 Training: GRPO with Unsloth

**Model:** Qwen2.5-7B-Instruct (4-bit QLoRA via Unsloth)  
**Algorithm:** Group Relative Policy Optimization (GRPO)  
**Training file:** `colab_training.ipynb` (Google Colab, T4 GPU)

### What GRPO does
For each observation prompt, it generates **4 completions** (different JSON actions), runs each in the environment, gets rewards, then updates the model to increase probability of higher-reward completions relative to the group average.

```
Prompt (observation) → [action_1, action_2, action_3, action_4]
                              ↓         ↓         ↓         ↓
                           r=0.20    r=-0.05   r=0.50   r=-0.50
                                              ↑ winner
GRPO update: increase P(action_3) relative to group mean
```

### What the agent learns
- Start with `inspect_metrics({})` to get overview (not random)
- Follow the topology: gateway → api → auth (upstream = root cause)
- Run `run_security_scan` before isolating services
- Ignore false alerts (the cache in hard task is a decoy)
- Submit exact diagnosis labels

---

## 🎨 Dashboard Changes (Round 2 Retheme)

The dashboard was redesigned from **cyberpunk neon** to **clean minimal professional**:

| Before | After |
|--------|-------|
| Dark background `#050811` | Light background `#f8fafc` |
| Neon cyan/green glows | Clean blue `#2563eb`, green `#16a34a`, red `#dc2626` |
| Animated grid background | Clean white cards with subtle shadows |
| `alert()` for comparison hint | Non-blocking toast notification |

### Bugs Fixed (10 total)
1. ✅ Topology nodes now highlight in red when they're affected services
2. ✅ Log stream clears properly on episode reset
3. ✅ `alert()` replaced with `showToast()` (non-blocking)
4. ✅ Progress bar colors are now dynamic (green/orange/red based on score)
5. ✅ Score total color resets to neutral on UI reset
6. ✅ `comparisonScores` preserved across `resetUI()` so Compare works after replay
7. ✅ Latency bar scale fixed (was /5000, now /2000 for better visibility)
8. ✅ Chart theme updated to match clean palette
9. ✅ Panel backgrounds consistent across all three columns
10. ✅ Topology node `topo-node.affected` CSS was always defined but JS never applied the class — fixed

---

## 📦 Files Overview

```
opensecops_env/
├── env.py              # Core environment (reset/step/state)
├── multi_agent_env.py  # ← NEW: Red vs Blue adversarial environment
├── curriculum.py       # ← NEW: Procedural task generation & level ups
├── models.py           # SecOpsAction, SecOpsObservation, SecOpsState
├── grader.py           # Deterministic episode grader [0,1]
├── tasks/
│   └── task_definitions.py  # 4 base task configs
└── server/
    └── app.py          # FastAPI server + endpoints + dashboard

training/
├── train_grpo.py       # GRPO training script (local)
└── plot_rewards.py     # Reward curve generation

colab_training.ipynb    # Full Colab notebook for HF credits
inference.py            # Baseline heuristic agent
```

---

## 🏆 Round 2 Theme Alignment

| Theme | Status | Evidence |
|-------|--------|---------|
| **Long-Horizon Planning** | ✅ Strong | 30–50 step episodes, sparse final reward, causal chain reasoning |
| **World Modeling: Professional** | ✅ Strong | Real SecOps tools (block_ip, rollback_deployment), realistic metrics |
| **Multi-Agent** | ✅ Strong | **Red vs Blue**: An attacker (Red) interleaves turns with the defender (Blue). Red actively obfuscates logs, spikes metrics, and injects false alerts while Blue investigates. |
| **Self-Improvement** | ✅ Strong | **Curriculum Manager**: Environment dynamically scales difficulty. When agent scores > 0.72 consistently, it unlocks Level 2 (compound failures), Level 3 (unknown procedurally generated topologies), and Level 4 (Adversarial Red). |

---

## 🚀 How to Execute & Test

### 1. Test the API Endpoints (Curriculum & Multi-Agent)
Start the server locally:
```bash
uvicorn opensecops_env.server.app:app --reload
```

In another terminal, test the Curriculum endpoint:
```bash
curl http://localhost:8000/curriculum/summary
# You will see the current level (1) and performance stats.
```

Test the Multi-Agent environment (Blue step, then Red step):
```bash
# Reset with "auto" to use curriculum
curl -X POST http://localhost:8000/multi/reset \
  -H "Content-Type: application/json" -d '{"task_id":"auto"}'

# Blue takes a step
curl -X POST http://localhost:8000/multi/blue/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"inspect_metrics", "parameters":{}}'

# Red reacts (heuristic attacker)
curl -X POST http://localhost:8000/multi/red/step \
  -H "Content-Type: application/json" -d '{}'
```

### 2. View the Dashboard
Go to `http://localhost:8000/dashboard` in your browser.
Click **Run Episode** and you will see the new clean, minimal theme, functional topology highlighting, and fixed progress bars.

### 3. Run the Colab Notebook (The Core Training Evidence)
1. Upload `colab_training.ipynb` to Google Colab.
2. Select a **T4 GPU** runtime (or A100 if using HF credits).
3. Add your `HF_TOKEN` in Colab's Secrets tab.
4. Run all cells. It will:
   - Download the model and environment
   - Run GRPO training using Unsloth
   - Automatically generate `training_results.png` (learning curves)
   - Push the fine-tuned adapter to Hugging Face.

**Primary pitch to judges:** This environment isn't just a static puzzle. It actively fights back (Multi-Agent) and gets harder as the model learns (Self-Improvement), making it a true testbed for next-generation reasoning models.
