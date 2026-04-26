# OpenSecOpsEnv — Full Technical Analysis
## How Everything Actually Works + Theme Alignment

---

## The Honest Truth: Is It Learning During the Demo?

**Short answer: No — not in real-time. The learning already happened.**

Here is the exact distinction:

| Phase | When | What changes |
|-------|------|-------------|
| **GRPO Training** | Happened on A100 GPU in HF Spaces | Model WEIGHTS updated. This is real, permanent learning. |
| **Inference (Demo)** | Happening right now on the dashboard | Model weights FROZEN. The model applies what it learned. |
| **Curriculum Tracking** | Happening live during your session | Task difficulty increases. Weights do NOT change. |

The model is NOT currently learning from dashboard episodes. What you see in the Learning tab is the **curriculum difficulty progressing** (harder scenarios unlocking) — not weight updates. This is by design.

However — there is deep, real learning in the GRPO phase. That's the part that matters.

---

## Full Pipeline — Agent Tab (Single Agent)

```
User clicks "Run Episode"
         │
         ▼
Browser opens SSE connection → /demo/stream?task_id=hard_data_exfiltration&mode=trained
         │
         ▼
Server: OpenSecOpsEnv.reset("hard_data_exfiltration")
    ├── Loads TASK_HARD config (fixed metrics, alerts, logs, topology)
    ├── Injects random seed jitter (±8% on CPU/mem/latency) → different each run
    └── Sends "reset" SSE event → dashboard renders System State panel
         │
         ▼
         ┌─────────────────── EPISODE LOOP ────────────────────────┐
         │                                                           │
         │  ① Server builds observation text (alerts, metrics, logs)│
         │                                                           │
         │  ② POST → HF Inference Endpoint                          │
         │     Payload: {                                            │
         │       "inputs": "=== Step 3 ===\nALERTS:\n [CRITICAL]...",│
         │       "parameters": { "max_new_tokens": 128,             │
         │                       "temperature": 0.3 }               │
         │     }                                                     │
         │                                                           │
         │  ③ Qwen2.5-7B-GRPO generates action JSON:               │
         │     {"action": "INVESTIGATE", "details": {"hosts":["db"]}}│
         │                                                           │
         │  ④ _parse_ai_action() maps → SecOpsAction                │
         │     action_type="query_logs", parameters={"service":"db"} │
         │                                                           │
         │  ⑤ env.step(action) → new observation, reward, done      │
         │     Reward logic:                                         │
         │       +1.0 correct diagnosis (submit_diagnosis)           │
         │       +0.5 correct mitigation (isolate_service)           │
         │       +0.3 correct investigation (query right service)    │
         │        0.0 neutral (inspect healthy service)              │
         │       -0.3 wrong service                                  │
         │       -1.0 wrong diagnosis                                │
         │                                                           │
         │  ⑥ Server sends "step" SSE event →                       │
         │     dashboard updates action card, system state, chart    │
         │                                                           │
         │  Repeat until done=True (submit_diagnosis) or max_steps  │
         └───────────────────────────────────────────────────────────┘
         │
         ▼
Server: grade(env.state.to_dict()) → GradeResult
    ├── diagnosis_correct: 1.0 if right label, 0.0 if wrong
    ├── action_efficiency: 1 - (wasted_steps / total_steps)
    └── investigation_quality: did it scan/query before diagnosing?
         │
         ▼
_curriculum.record_score(task_id, score) → level-up check
         │
         ▼
Server sends "grade" SSE event → dashboard shows final scores + episode end modal
```

**The 4 incident types and what the AI must figure out:**

| Scenario | Ground truth | Noise/traps |
|----------|-------------|------------|
| Memory Leak | `auth` service leaking RAM → restart it | 5% noise. Easy to spot. |
| DDoS Cascade | External IPs flooding gateway → block them + scale api | 25% noise. 2 attack IPs. |
| Bad Deployment | api v2.4.1 pushed bad Redis config → rollback | 35% noise. Cache looks like hardware. |
| Data Exfiltration | `reports_bot` account exfiltrating 4GB → isolate db + block IP | **55% noise + false cache CRITICAL alert the attacker injected** |

**What is PARTIAL OBSERVABILITY?**
The AI never sees the "hidden state" (what the real root cause is, what the attack_progress number is, which services are actually affected). It only sees: alerts, metrics visible in logs, and action results — exactly like a real engineer.

---

## Full Pipeline — Battle Mode (Multi-Agent)

```
User clicks "Start Battle"
         │
         ▼
Browser opens SSE → /battle/stream?task_id=hard_data_exfiltration
         │
         ▼
Server: MultiAgentSecOpsEnv.reset("hard_data_exfiltration")
    ├── Creates OpenSecOpsEnv (shared game world)
    ├── Creates RedAgentState (attacker: round=0, damage=0)
    ├── Applies random seed → different starting conditions
    └── Sends "battle_reset" SSE event
         │
         ▼
         ┌─────────────── BATTLE ROUND LOOP ───────────────────────┐
         │                                                           │
         │  🔴 RED TURN (Attacker — heuristic AI):                 │
         │     _heuristic_red_action() picks based on game state:   │
         │       IF attack_progress < 0.8 → amplify_attack          │
         │       ELIF healthy services exist AND rand < 0.4          │
         │                             → corrupt_metric             │
         │       ELIF cyber_attack AND rand < 0.3 → accelerate_spread│
         │       ELSE → inject_noise                                 │
         │                                                           │
         │     red_step() mutates the environment:                  │
         │       amplify_attack: hidden.attack_progress += 0.05-0.15│
         │       corrupt_metric: spikes healthy svc cpu+latency      │
         │       create_false_alert: adds fake CRITICAL in alert list│
         │       accelerate_spread: adds new svc to affected_services│
         │       inject_noise: adds misleading log entry             │
         │                                                           │
         │     Server sends "red_step" SSE → attacker card shown    │
         │                                                           │
         │  🔵 BLUE TURN (Defender — your trained AI):              │
         │     [Same as Agent pipeline — POST to HF endpoint]        │
         │     AI sees polluted observation (attacker's noise is IN) │
         │     Must untangle real signals from attacker's fakery     │
         │                                                           │
         │     ma_env.blue_step(action) → env.step(action)          │
         │     Server sends "blue_step" SSE → defender card shown    │
         │                                                           │
         │  Repeat until done=True                                   │
         └───────────────────────────────────────────────────────────┘
         │
         ▼
grade() → same scoring as Agent tab
winner = "defender" if blue_cumulative > red_cumulative
_curriculum.record_score(task_id, score)
Server sends "battle_end" SSE → battle result modal
```

**What is the point of Battle Mode?**

> It answers the question: "Can your trained model perform under active adversarial conditions?"

A model trained in a clean environment might be fooled by:
- Fake CRITICAL alerts on the wrong service
- Corrupted metrics that make healthy services look sick
- Log noise designed to look like a memory leak when it's actually an exfiltration

Watching the trained AI correctly isolate `db` and block `10.0.0.99` EVEN WHEN the attacker has injected a fake cache alert is a concrete demonstration that the model learned meaningful representations, not pattern matching.

**What score does what:**

| Score element | Battle context |
|---------------|---------------|
| **Defender cumulative** | Sum of reward from Blue's actions. High = targeted, correct actions. |
| **Attacker cumulative** | Sum of Red's damage-dealing. High = Red successfully confused the defender. |
| **Defender Advantage** | Blue minus Red. Positive = your AI is winning. |
| **Attack Suppression** | % of Red's maximum possible damage that Blue neutralised. 76% = very strong defence. |
| **Episode Score** | The OpenEnv grader's independent assessment of final state. Unaffected by who "won" the battle. |

---

## Full Pipeline — Learning Tab

```
After each episode completes (grade event):
         │
         ▼
_curriculum.record_score(task_id, score)
    └── Appends {episode, task_id, score, level} to score_history
    └── Checks rolling window of last 5 scores
    └── If avg >= threshold for current level → current_level += 1
         │
         ▼
Browser: GET /curriculum/summary
    └── Returns {current_level, total_episodes, level_up_history}
         │
         ▼
Dashboard: renderCurriculum(data) → updates left panel + level progress bar
Dashboard: improvementChart.update() → updates the score/level chart

Level-Up Thresholds:
  Level 1 → 2: avg ≥ 0.65 over 5 episodes
  Level 2 → 3: avg ≥ 0.70
  Level 3 → 4: avg ≥ 0.72
  Level 4 → 5: avg ≥ 0.75
```

**Why does the left panel say "No episodes recorded"?**

The curriculum manager lives in SERVER memory. When the server restarts, it resets to 0. The chart (right panel) uses BROWSER memory — so it shows episodes from the current browser session. This is the disconnect you see. After running episodes without restarting the server, both panels stay in sync.

**Is the Learning Tab faking the chart?**

NO. After 8 real episodes in the screenshots, the chart shows real scores ranging from ~0.10 to ~0.75. The fluctuation is real — some episodes the AI makes good choices (high score), some it gets confused by attacker noise or runs out of steps (low score). The variance is honest.

---

## Theme Alignment Analysis

### Theme #1 — Multi-Agent Interactions ✅ STRONG FIT

**How it fits:**
- Battle Mode is a live competitive multi-agent game: Red (Attacker) vs Blue (Defender)
- Both agents perceive the SAME partially observable environment — partial observability by design
- Red's actions affect what Blue sees (corrupted metrics, false alerts) — theory-of-mind is required
- Blue must reason about WHY the alerts look wrong (attacker behaviour modelling)

**What we have:**
- Red agent: heuristic policy with strategic action selection (prefers amplify → spread → corrupt → noise)
- Blue agent: LLM (Qwen2.5-7B-GRPO) with full contextual reasoning
- Adversarial interference: Red can add fake alerts, corrupt metrics, spread attack
- Joint reward structure: Red maximises damage, Blue maximises resolution

**Gap to acknowledge:**
- The Red agent does not use an LLM — it uses a heuristic. This means the multi-agent learning is asymmetric: only Blue is the trained policy. Red's strategy is fixed.
- Future work: train a Red LLM policy via self-play against Blue, creating true emergent strategy

**What to say to judges:**
> "Battle Mode implements competitive multi-agent interaction in a partially observable security environment. The Blue defender must model attacker behaviour to distinguish injected false alerts from real signals — this requires implicit theory-of-mind. The trained model outperforms the heuristic baseline even under active adversarial interference."

---

### Theme #4 — Self-Improvement ✅ STRONG FIT

**How it fits:**
- The GRPO training algorithm IS self-play based self-improvement:
  - Generate N candidate action sequences for same observation
  - Rank by environment reward (GRPO group scoring)
  - Reinforce better sequences, suppress worse ones
  - Repeat → model learns to generate its own better responses
- The Curriculum system drives adaptive difficulty: agent must master easy scenarios before hard ones unlock
- The reward signal is environment-derived (not human labelled) — the environment teaches the agent

**What we have:**
- GRPO training: 500 steps on A100, reward-based group ranking → real weight updates
- Curriculum: 5 levels, auto-advancing based on rolling average score
- Score variance: the fluctuating chart in Learning tab is genuine performance variation

**Gap to acknowledge:**
- The curriculum doesn't update model weights — it only selects which task to present. Weight updates would require an online training loop (inference → env → reward → gradient step), which requires persistent GPU compute.
- What we demonstrate is the TRAINING side of self-improvement (the trained model), and the TASK SELECTION side (curriculum) — not online weight updates during the demo.

**What to say to judges:**
> "Self-improvement happens at two levels: during GRPO training offline (the model improves its own action quality through reward-ranked group comparison — the DeepSeek-R1 approach) and during deployment via adaptive curriculum (the system selects harder scenarios as the agent proves competence). The Learning tab shows this trajectory across episodes."

---

### Theme #3.1 — World Modeling / Professional Tasks ✅ STRONG FIT

**This is actually your strongest theme alignment.**

**How it fits:**
- The environment models a real professional SecOps workflow: alerts → investigate → diagnose → mitigate
- Partial observability: the agent cannot directly read the hidden state (root cause, attack progress)
- Tool interaction: each action is a real tool call (query logs, run security scan, block IP)
- Multi-step workflow: investigation → mitigation → diagnosis is a realistic 8-15 step sequence
- Causal reasoning required: blocking the wrong IP has no effect; blocking the right one stops the attack
- State persistence: actions have persistent consequences (isolated services stay isolated)

**Specifically for Data Exfiltration:**
- The agent must reason causally: "high cache memory" is a RED INJECTED false alert, "unusual outbound db traffic" is real
- This requires understanding service topology: reports_bot → db → external_ip is the attack chain
- The agent must resist the mislead and focus on the real signal

---

### Theme #2 — Long-Horizon Planning ❌ PARTIAL FIT

**Where it fits:**
- Multi-step reasoning with delayed rewards: investigate → mitigate → diagnose spans 8-30 steps
- Agent must track state across the episode (what did it already scan? what logs did it see?)
- Early mistakes compound: isolating the wrong service causes the right one to keep exfiltrating

**Where it doesn't fully fit:**
- Max episode length is 30 steps — the theme targets "very extended" trajectories and sessions beyond context memory limits
- The environment doesn't require decomposing complex high-level goals into sub-goals

**What to say to judges:**
> "While not our primary theme target, OpenSecOpsEnv involves structured multi-step planning with sparse rewards — the model must reason across 10-30 steps before the terminal diagnosis action, which tests durable internal representations."

---

## What "Is It Learning?" Actually Means

### During Training (✅ Happened — real learning)

```
For each training batch:
  1. Generate 4 candidate responses for the same observation (GRPO group)
  2. Run each response through env.step() → get rewards
  3. Score: group-relative advantage = (reward_i - mean_group) / std_group
  4. Policy gradient update: reinforce high-advantage responses
  5. Repeat 500 times → weights shift toward actions that earn +reward
```

The model's weights changed. This is real machine learning.

### During Curriculum (Demonstration, not learning)

```
Episode n completes:
  score = grade(env.state)   # 0.0 to 1.0
  history[-5:] average > threshold?
    YES → task pool expands to include next difficulty level
    NO  → stay at current level, keep practicing
```

No weights change. The task selection policy adapts, not the model.

### The Visual on the Learning Tab

The chart shows **episode scores over time** within your current server session. Fluctuation (0.1 → 0.7 → 0.4 → 0.75) is NORMAL and real because:
- Each episode has a random seed → different starting metric values
- The AI model has temperature > 0 → different action choices
- The attacker's heuristic has randomness → different interference patterns

A flat 0.99 every episode would actually be SUSPICIOUS (it would mean the model isn't making real decisions).

---

## Summary for Judges

| What you built | Fits which theme |
|----------------|-----------------|
| OpenSecOpsEnv environment | Theme #3.1 — Professional World Model |
| GRPO training on environment rewards | Theme #4 — Self-Improvement (training) |
| Adaptive curriculum (5 levels) | Theme #4 — Self-Improvement (task selection) |
| Red vs Blue Battle Mode | Theme #1 — Multi-Agent Interactions |
| Multi-step investigation episodes | Theme #2 — Long-Horizon Planning (partial) |
| Live AI inference on dashboard | All themes — demonstration |
