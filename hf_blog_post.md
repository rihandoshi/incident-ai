# Teaching AI to Be a Security Engineer — from 0.22 to 0.76 on the Hardest Attack Scenario

*How we built an adversarial, multi-agent SecOps incident response environment and fine-tuned Qwen2.5-7B with GRPO to investigate disguised cyberattacks — while a live Red Agent actively tries to sabotage it.*

---

## The Problem

It's 3AM. Your phone rings. PagerDuty is blazing.

```
[CRITICAL] cache · memory 82% — triggering high-memory threshold policy
[WARNING]  db · CPU 82% — elevated for time of day
[CRITICAL] auth · memory 89% — approaching limit
[WARNING]  db · Outbound traffic from DB to external IP — unusual
```

Four alerts. Three of them are **fake** — planted by the attacker who is currently exfiltrating your database. One alert is real, but it's buried in noise.

A junior engineer panics and restarts the cache service — the one with the fake alert. This takes down your checkout flow. The real exfiltration continues for 40 more minutes until someone notices 4GB of customer data is gone.

**Can a language model learn not to make that mistake?**

That is exactly the problem OpenSecOpsEnv is designed to answer.

---

## What We Built

**OpenSecOpsEnv** is a fully OpenEnv-compliant RL environment that simulates a distributed production system under attack. It is designed from first principles to be a hard, meaningful benchmark for LLM agent reasoning — not a toy.

### The System Being Defended

Five microservices: `gateway → api → auth`, `api → cache`, `api → db`, `auth → db`

Each episode simulates one of four real incident types with increasing complexity:

| # | Task | Difficulty | Noise | The Trap |
|---|------|-----------|-------|---------|
| 1 | Memory Leak in auth | Easy | 5% | No trap — straightforward |
| 2 | DDoS Cascade | Medium | 25% | Two attacker IPs, cascade through 3 services |
| 3 | Bad Deployment | Med-Hard | 35% | Gateway alert misleads; real issue is Redis misconfiguration |
| 4 | **Data Exfiltration** | **Hard** | **55%** | **Attacker plants a fake CRITICAL cache alert; real attack is on db** |

### What "Partial Observability" Actually Means

The agent is **never told what the root cause is**. It sees exactly what a real on-call engineer sees:

- Metric snapshots (CPU, memory, latency, error rate per service)
- Log lines (up to 8 per step, selected with noise)
- Active alerts (some real, some Red-Agent-planted fakes)
- Service topology (who talks to who)
- The result of its last action

The hidden state (true root cause, attack progress, which alerts are fake) is invisible to the agent. It must correlate observations across steps and services.

### The 9-Action API

Every action the agent takes is a structured JSON call:

```json
{"action_type": "run_security_scan", "parameters": {"target": "db"}}
{"action_type": "isolate_service",   "parameters": {"service": "db"}}
{"action_type": "block_ip",          "parameters": {"ip": "10.0.0.99"}}
{"action_type": "submit_diagnosis",  "parameters": {"label": "cyber_attack:data_exfiltration"}}
```

**Wrong actions have real consequences:**
- `restart_service` on the wrong service: causes unnecessary downtime (negative reward)
- `isolate_service` on a healthy service: takes down a dependency (large negative)
- `submit_diagnosis` with wrong label: episode ends with score near zero

---

## The Adversarial Layer — Multi-Agent Battle Mode

Here is what makes OpenSecOpsEnv genuinely hard and different from other RL environments.

While the Blue Agent (your trained LLM) is investigating, a **Red Agent (Attacker) is simultaneously modifying the environment:**

| Red Action | What it does to the environment |
|-----------|--------------------------------|
| `amplify_attack` | Increases `hidden.attack_progress` by 0.05–0.15 |
| `corrupt_metric` | Spikes CPU/latency on a **healthy** service to create a false performance signal |
| `create_false_alert` | Appends a new CRITICAL alert on an unaffected service |
| `accelerate_spread` | Adds a new service to `affected_services` by spreading through topology |
| `inject_noise` | Adds misleading log entries like `[cache] WARN Memory spike (harmless GC event)` |

The Red Agent acts on **the same shared environment state** as the Blue Agent. This is true multi-agent interaction: the Blue Agent's observations are corrupted in real time by the Red Agent's actions.

**The result:** On the Data Exfiltration task, the Blue Agent receives a fake CRITICAL alert on `cache` AND an amplifying attack on `db`. It must reason: *"The cache alert looks bad but the real signal is in the db logs — the reports_bot service account is exfiltrating data."*

This is theory-of-mind in action: the agent must implicitly model an adversary's behaviour to distinguish signal from planted noise.

---

## The Reward Function

Dense rewards at **every single step** — not binary success/fail:

| Event | Reward | Design Rationale |
|-------|--------|-----------------|
| Query logs on affected service | +0.20 | Reward targeted investigation |
| Security scan on affected service | +0.30 | Reward hypothesis-driven scanning |
| Correct mitigation (right service) | +0.50 | Reward precise action |
| Correct final diagnosis | +1.00 | Maximum reward for getting it right |
| Irrelevant investigation | -0.05 | Small penalty to discourage scatter-gun |
| Ineffective mitigation | -0.10 | Penalise wasted actions |
| Harmful action (wrong service/IP) | -0.50 | Hard penalty for making things worse |
| Wrong final diagnosis | -1.00 | Maximum penalty for getting it wrong |
| Step cost | -0.02 | Per-step efficiency pressure |

**Why dense rewards matter for GRPO:** GRPO ranks groups of candidate responses by their rewards. Dense rewards mean every step produces a meaningful gradient signal — the model doesn't have to wait until the episode end to get feedback.

### Episode Grader (The Final Score)

At the end of each episode, the OpenEnv grader computes:

```
score = 0.5 × diagnosis_correct
      + 0.3 × action_efficiency  
      + 0.2 × investigation_quality
```

- `diagnosis_correct`: 1.0 if correct label, 0.0 otherwise
- `action_efficiency`: 1 - (wasted_steps / total_steps)
- `investigation_quality`: did the agent scan/query the right services before diagnosing?

---

## GRPO Training — What We Actually Did

We fine-tuned **Qwen2.5-7B-Instruct** using GRPO (Group Relative Policy Optimization) — the same algorithm used in DeepSeek-R1 — via Hugging Face TRL and Unsloth for 4-bit quantized training on an A100 GPU.

### Why GRPO for This Task?

Standard RLHF needs a reward MODEL. We don't have one — we have an ENVIRONMENT. GRPO directly uses environment rewards as the optimization signal, which is exactly the right fit.

```python
def secops_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        action = parse_action(completion)       # Parse JSON from LLM output
        if action is None:
            rewards.append(-0.5)               # JSON format penalty
            continue
        env = OpenSecOpsEnv()
        env.reset(task_id)
        _, reward, _, _ = env.step(action)      # Execute in environment
        rewards.append(float(reward) - 0.02)   # Apply step cost
    return rewards
```

**GRPO mechanism:**
1. For each training example, generate **4 candidate action JSON responses**
2. Execute all 4 in the environment → get 4 rewards
3. Compute group-relative advantage: `A_i = (r_i - mean(group)) / std(group)`
4. Policy gradient update: reinforce high-advantage responses, suppress low ones
5. Repeat 500 steps

### Training Setup

```python
training_args = GRPOConfig(
    num_generations=4,       # 4 candidates per observation
    max_new_tokens=128,      # Enough for JSON action + reasoning
    temperature=0.9,         # Exploration during training
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)
```

LoRA adapters: `r=16`, targeting all attention + MLP projections. Merged to 16-bit for deployment (no bitsandbytes dependency in production).

---

## Results

### Training Curves

![Training Results — Reward curve, GRPO loss curve, Before vs After](./training_results.png)
*Left: Per-step reward rising from random baseline (~0.10) to trained level (~0.20). Centre: GRPO policy loss rising = gradient actively updating weights from reward differences (this is correct, healthy GRPO behaviour). Right: Episode scores across all 4 task difficulties after 500 training steps.*

**Reward curve:** Average per-step reward rises from ~0.10 (random baseline) to a stable plateau of ~0.18–0.20. The per-step average is naturally bounded — most steps are neutral investigations (+0.0). The large rewards (+1.0 correct diagnosis, -1.0 wrong) only fire occasionally, so episode-level performance is the more meaningful metric.

**Loss curve:** In GRPO, **rising policy loss is the signal that learning is happening** — not a problem. Loss near zero means the model is not differentiating between candidate responses. Rising loss from step ~100 onward confirms the policy gradient is finding meaningful reward differences and updating weights. This is exactly the expected training signature.

### Before vs After (Episode Score [0, 1])

![Training Results](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo/resolve/main/training_results.png)

| Task | Difficulty | Untrained | After GRPO | Improvement |
|------|-----------|-----------|------------|-------------|
| Memory Leak | Easy | 0.51 | **0.95** | **+86%** |
| DDoS Cascade | Medium | 0.35 | **0.87** | **+149%** |
| Bad Deployment | Med-Hard | 0.31 | **0.81** | **+161%** |
| Data Exfiltration | Hard | 0.22 | **0.76** | **+245%** |

**Key finding:** The hardest task shows the most dramatic improvement. The untrained model essentially guesses on Data Exfiltration (0.22 ≈ random). After 500 GRPO steps, the trained model reaches 0.76 — reliably:
- Ignoring the fake CRITICAL cache alert
- Querying `db` and `auth` logs
- Running a security scan on `db`
- Identifying `reports_bot` as the compromised account
- Isolating `db` and blocking `10.0.0.99`
- Submitting `cyber_attack:data_exfiltration` as the correct diagnosis

**This is not pattern-matching** — each episode has a randomly seeded starting state with metric jitter, so the model cannot memorise a fixed sequence.

---

## The Self-Improvement Curriculum

Beyond single episodes, we implemented a 5-level curriculum that automatically advances difficulty as the agent scores well:

```
Level 1: easy_memory_leak              (threshold: avg ≥ 0.65 over 5 episodes)
Level 2: + medium_ddos_cascade         (threshold: avg ≥ 0.70)
Level 3: + medium_hard_bad_deployment  (threshold: avg ≥ 0.72)
Level 4: + hard_data_exfiltration      (threshold: avg ≥ 0.75)
Level 5: hard_data_exfiltration only   (maximum difficulty)
```

This is the Self-Improvement theme: the agent earns its way to harder challenges by demonstrating capability. It cannot skip to the hardest task — it must prove competence at each level.

The curriculum is tracked live on the dashboard's Learning tab.

---

## The Live Demo Dashboard

We built a full real-time dashboard that streams live AI reasoning to the browser via Server-Sent Events (SSE):

- **Agent Tab:** Watch Qwen2.5-7B-GRPO investigate and resolve incidents in real time. Every action card shows the raw JSON the model generated — not a pre-scripted replay.
- **Battle Tab:** Live Red vs Blue stream. Watch the attacker inject fake alerts mid-episode and watch the trained model resist the mislead.
- **Learning Tab:** Live curriculum progress. As you run episodes, the score history charts and level-up events populate in real time.

The "🤖 AI Output" box on every action card is the actual model output from the HF Inference Endpoint — verifiable proof of live inference.

---

## Theme Alignment

| Theme | How We Fit |
|-------|-----------|
| **#1 Multi-Agent** | Battle Mode: Red Attacker mutates environment in real-time while Blue Defender (LLM) must reason about the adversary's interference |
| **#4 Self-Improvement** | GRPO training is reward-ranked group self-improvement; curriculum advances difficulty based on demonstrated competence |
| **#3.1 World Modeling** | Full professional SecOps workflow with tool use, persistent state, partial observability, and causal action consequences |

---

## Links

| Resource | URL |
|----------|-----|
| 🤗 Trained Model | [SapphireGaze429/opensecops-qwen2.5-7b-grpo](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo) |
| 🎮 Live Demo Dashboard | [HF Space](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training) |
| 📓 Training Notebook | [colab_training.ipynb](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training/blob/main/colab_training.ipynb) |
| 📊 Training Plots | [training_results.png](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo/resolve/main/training_results.png) |
| 📖 GitHub | [incident-ai](https://github.com/SapphireGaze429/incident-ai) |

---

*Built for the OpenEnv Hackathon Round 2. Fully compliant with the OpenEnv API (reset/step/state/grade). No mock data. No pre-scripted replays. Every dashboard action is live model inference.*
