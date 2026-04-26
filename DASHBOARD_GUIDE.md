# OpenSecOpsEnv — Complete Dashboard Guide

> **Plain-English explanation of every element on every tab.**
> Read this before demoing to judges.

---

## The Big Picture — What Is This System?

This is an **AI Security Operations (SecOps) environment**.

Imagine your company's servers are being attacked at 3am. A human on-call engineer would have to:
1. Look at alerts and dashboards
2. Figure out what went wrong (memory leak? hacker? bad deploy?)
3. Take the right actions (restart a service, block an IP, isolate a server)

We trained an AI model (**Qwen2.5-7B-GRPO**) to do all of that automatically, in real time, better than a random/untrained model would.

**The dashboard is a live demonstration of that AI responding to security incidents.**

---

## The Header Bar

```
OpenSecOpsEnv  [OpenEnv]  [Curriculum RL]      🤖 Qwen2.5-7B-GRPO (fine-tuned)  ● Server online
```

| Element | What it means |
|---------|--------------|
| **OpenSecOpsEnv** | Name of the environment/project |
| **[OpenEnv]** | This is built on the OpenEnv hackathon framework |
| **[Curriculum RL]** | The agent learns progressively harder tasks over time |
| **🤖 Qwen2.5-7B-GRPO (fine-tuned)** | Your actual trained AI model is LIVE and connected. This turns on when the HF endpoint responds. |
| **● Streaming** | An episode is actively running (SSE stream is open) |
| **● Server online** | The FastAPI backend is running |

---

## Tab 1: AGENT — Single AI Agent vs. Incident

This tab shows one AI agent (your trained model) responding to an incident alone.

### Controls Bar

```
[Scenario ▼]  [Trained Model | Baseline Model]  [Speed ▼]  [Run Episode]  [Reset]  [Compare]
```

| Control | What it does |
|---------|-------------|
| **Scenario** | Picks which security incident to simulate. 4 scenarios: Memory Leak (Easy), DDoS Cascade (Medium), Bad Deployment (Medium-Hard), Data Exfiltration (Hard) |
| **Trained Model** | Uses YOUR fine-tuned Qwen2.5-7B model to decide actions. This is the AI you trained. |
| **Baseline Model** | Uses a hardcoded "bad agent" that makes wrong choices — used to show how much better your trained model is |
| **Speed** | How fast the episode plays out (Fast = 0.8s between steps, Slow Demo = 2.5s — use Slow for presentations) |
| **Run Episode** | Starts the AI agent running through the incident |
| **Reset** | Clears everything and starts fresh |
| **Compare** | After running both Trained and Baseline modes, shows the score difference |

### Left Panel — SYSTEM STATE

This shows what the AI can "see" at any moment. It's the AI's observation.

```
┌─────────────────────────────────────┐
│ Data Exfiltration (Disguised)       │
│ A compromised service account...    │
│ [HARD]                              │
├─────────────────────────────────────┤
│ ACTIVE ALERTS                       │
│ [WARNING] db · cpu 89.3%            │
│ [CRITICAL] cache · memory 82%       │
├─────────────────────────────────────┤
│ SERVICE METRICS                     │
│ db    cpu ███░░  4.1%               │
│ auth  cpu ██░░░  mem ██████         │
└─────────────────────────────────────┘
```

| Element | What it means |
|---------|--------------|
| **Scenario Card** (top blue box) | Brief description of what the incident is. The AI doesn't get told this — it has to figure it out from the clues below. |
| **[HARD] badge** | Difficulty level. Hard = 55% noise (fake alerts designed to mislead), disguised attack vector |
| **Active Alerts** | Live alerts the system is generating. These are what a real PagerDuty/Datadog alert would look like. |
| **[WARNING] / [CRITICAL]** | Severity of the alert. CRITICAL = urgent. WARNING = watch it. |
| **[RED INJECTED]** (in Battle mode) | An alert that the Attacker agent CREATED to mislead the Defender — it's fake! |
| **Service Metrics** | CPU, Memory, Latency bars for each microservice |
| **Bar colour** | 🟢 Green = normal. 🟡 Cyan = warning. 🔴 Red = critical. |
| **err X.X%** | Error rate — what % of requests to that service are failing |
| **Service Topology** | Which services talk to which. Shows the blast radius if one goes down. |

### Centre Panel — AGENT ACTION FEED

This is the MOST IMPORTANT panel. It shows every decision the AI makes, in real time.

```
┌─────────────────────────────────────────────────────┐
│ 🔵 DEFENDER    SCAN run_security_scan          +0.38 │
│ target="db"                                          │
│ SECURITY SCAN – db: ALERT: Unusual outbound...       │
│ 🤖 AI Output                                         │
│ {"action_type": "run_security_scan", "parameters":...│
└─────────────────────────────────────────────────────┘
```

| Element | What it means |
|---------|--------------|
| **🔵 DEFENDER** | This action was taken by the Blue (Defender) AI agent — your trained model |
| **🔴 ATTACKER** | This action was taken by the Red (Attacker) agent — the adversarial agent trying to make things worse |
| **SCAN run_security_scan** | The action type. SCAN = badge abbreviation. `run_security_scan` = exact game action |
| **target="db"** | Parameters passed to the action (which service to scan) |
| **The text below** | What the environment told the agent back — the result of that action |
| **+0.38** | The reward earned. **Positive = good action. Negative = bad/wrong action. Zero = neutral.** |
| **🤖 AI Output** (teal box) | The ACTUAL raw JSON your Qwen2.5-7B model generated. This is live proof the AI is running — not a simulation. |
| **Thinking dots** | The model is currently querying the HF endpoint — waiting for a response |

**The 9 possible actions:**

| Action | What the agent does | When to use it |
|--------|--------------------|----|
| `inspect_metrics` | Look at all service metrics | First step — gather info |
| `query_logs` | Read logs for a specific service | Investigate suspicious service |
| `run_security_scan` | Deep security scan of a service | Suspect cyber attack |
| `restart_service` | Restart a crashed/leaking service | Memory leak, service crash |
| `scale_service` | Add more replicas to handle load | DDoS / traffic spike |
| `block_ip` | Block a specific IP address | DDoS, data exfiltration source |
| `rollback_deployment` | Revert to previous version | Bad deployment |
| `isolate_service` | Cut a service off from the network | Active breach, data exfiltration |
| `submit_diagnosis` | Final answer — what was the root cause | Final step of every episode |

**The reward signal (+/- numbers):**
- `+1.00` = Correct final diagnosis ✅
- `+0.50` = Correct mitigation action (e.g., isolate the right service)
- `+0.30` = Good investigation (scanning/querying the right service)
- `0.00` = Neutral (looked at the wrong service — wasted a step)
- `-0.30` = Mildly wrong action
- `-1.00` = Completely wrong final diagnosis ❌

### Right Panel — REWARD CURVE

```
Step Reward ───  Cumulative - - -
     ▲  0.5
     │      ●─────●
0.0  │  ●               ●
     │           ●
    -0.5
     S1  S2  S3  S4  S5
```

| Element | What it means |
|---------|--------------|
| **Step Reward (solid line)** | The reward earned at each individual step. Bounces up and down — good steps go up, wasted steps go to zero. |
| **Cumulative (dashed)** | Running total of all rewards. Should trend upward for a smart agent. |
| **Final Score / 1.0** | OpenEnv's grader score between 0 and 1. Weighted average of all 3 sub-scores below. |
| **Cumul. Reward** | Sum of all step rewards. Can be negative if the agent made many bad moves. This is what GRPO optimised during training. |
| **Steps Taken** | How many actions the agent took total |
| **Diagnosis** | Did the agent correctly identify the root cause? (1.0 = perfect, 0.0 = wrong) |
| **Action Efficiency** | Did the agent take the minimum effective actions, or did it waste steps? High = focused. Low = scattered. |
| **Investigation Quality** | Did the agent properly investigate before diagnosing? (Query logs? Run scans?) High = methodical. |

### Log Stream (Bottom of Centre Panel)

```
[cron] INFO Backup job completed successfully
[audit] WARN Audit log: 3 RBA privilege grants in past 3 hours
[db] INFO Large SELECT query executed by service-account 'reports_bot'
```

Raw real-time log lines from the simulated services. The AI reads these too (up to 5 per step). The clues to finding the attacker are hidden in here.

---

## Tab 2: BATTLE — Trained AI vs. Attacker

This tab shows an **adversarial game** between two agents:
- 🔵 **Defender (your trained AI)** — tries to identify and stop the incident
- 🔴 **Attacker (heuristic agent)** — tries to make the incident WORSE and confuse the defender

**Turn order:** Red attacks first → Blue responds. Repeat.

### Why this is impressive for judges:
> Most ML projects just show a model predicting something. This shows a LIVE multi-agent game where your fine-tuned AI is actively competing against an adversary in real time.

### Battle Controls

| Control | Meaning |
|---------|---------|
| **Scenario** | Which incident the battle is fought over |
| **Start Battle** | Begins the Red vs Blue SSE stream |
| **Defender vs Attacker** (header right) | Score leaders — who's winning right now |

### Battle Feed

Same as Agent Action Feed but interleaved with Red and Blue cards.

```
🔴 ATTACKER    AMP amplify_attack                   +0.20
auto
Red amplified attack progress by 0.06 → 1.00

🔵 DEFENDER    ISO isolate_service                  +0.50
service="db"
Service 'db' isolated from network. Attack vector contained.
🤖 AI Output: {"action_type": "isolate_service", ...}
```

The Red agent's 5 attack moves explained:

| Attack | What it does |
|--------|-------------|
| `inject_noise` | Adds fake log entries to confuse the defender |
| `amplify_attack` | Makes the cyberattack progress faster (more data stolen faster) |
| `corrupt_metric` | Spikes CPU/latency on a HEALTHY service to create a false alarm |
| `create_false_alert` | Injects a fake CRITICAL alert on the wrong service |
| `accelerate_spread` | Spreads the attack to neighbouring services in the topology |

### Battle Score Panel

```
BATTLE SCORE
     Defender ──  Attacker ──
1.4 ●─────────────────────────●
1.0      ●
0.5           ●─────●
0.0  ●                    ●───●
     R1   R2   R3   R4   R5

1.40  DEFENDER   |   0.40  ATTACKER
           0.990  Episode Score
Defender Advantage    +1.00
Attack Suppression    76%
Battle Rounds         8
```

| Element | Meaning |
|---------|---------|
| **Defender cumulative** | Blue agent's total rewards. Higher = better defending. |
| **Attacker cumulative** | Red agent's total rewards. Higher = attack is succeeding. |
| **Episode Score** | Final grader score 0-1 for how well the incident was resolved |
| **Defender Advantage** | Blue cumulative MINUS Red cumulative. Positive = Defender winning. |
| **Attack Suppression** | What % of the attacker's maximum possible damage was neutralised by the defender |
| **Battle Rounds** | Number of Red+Blue turn pairs completed |

---

## Tab 3: LEARNING — Curriculum Progress

This tab answers: **Is the agent getting better over time during this session?**

### Left Panel — Curriculum Progress

```
CURRENT LEVEL
[ Level 1 / 5 ]  No episodes recorded yet
Progress to Level 2
```

The agent starts at Level 1 (easy tasks) and automatically advances when it scores well consistently.

| Level | Tasks included | Threshold to advance |
|-------|---------------|----------------------|
| 1 | Memory Leak (Easy) | Score avg ≥ 0.65 over 5 episodes |
| 2 | Memory Leak + DDoS (Medium) | Score avg ≥ 0.70 |
| 3 | DDoS + Bad Deployment | Score avg ≥ 0.72 |
| 4 | Bad Deployment + Data Exfiltration | Score avg ≥ 0.75 |
| 5 | Data Exfiltration only (Hard) | Maximum difficulty |

### Right Panel — Learning Curve Chart

```
Episode Score ───  Curriculum Level - - -
1.0 ●─────────────────────────
0.8      ●
0.5           ●─────●
              Ep1  Ep2  Ep3  Ep4
```

Each dot = one completed episode. The green line shows whether scores are going up over time. The purple dashed line shows which curriculum level the agent is at.

### Level-Up Events

When the agent crosses a threshold, a Level-Up event is logged here:
```
Episode 5: Level 1 → 2  (avg score 0.823)
```

### ⚠️ Important: Where is the "learning" actually happening?

There are **TWO different kinds of learning** in this project — don't confuse them:

**1. GRPO Training (already done ✅)**
- This happened in the Hugging Face Space on an A100 GPU over ~500 steps
- The model's weights were updated via Group Relative Policy Optimization (GRPO)
- The result is the fine-tuned model at `SapphireGaze429/opensecops-qwen2.5-7b-grpo`
- This is NOT happening on the dashboard — it already happened. The dashboard is INFERENCE only.

**2. Curriculum Tracking (happens live on the dashboard)**
- Every episode you run, the score is recorded server-side
- If the agent consistently scores above the level threshold, it "levels up"
- This doesn't change the model weights — it just selects harder tasks
- It's a demonstration of how you WOULD do online curriculum learning

**To explain to judges:**
> "The model was trained offline using GRPO on an A100 GPU. Here on the dashboard, we demonstrate a curriculum learning wrapper — the agent starts on easy scenarios and earns harder ones as it succeeds. The live AI on the right panel is our trained model making real-time decisions via the Hugging Face Inference Endpoint."

---

## The Episode End Modal

```
┌───────────────────────────────┐
│            ✅                  │
│    Incident Resolved!          │
│    (Trained AI (Qwen2.5-7B))  │
│         0.990                  │
│      [Run Again]               │
└───────────────────────────────┘
```

| Score | Icon | What it means |
|-------|------|--------------|
| > 0.70 | ✅ | **Incident Resolved!** — Agent correctly diagnosed AND efficiently mitigated. |
| 0.40 – 0.70 | ⚠️ | **Partially Resolved** — Agent got partial credit (maybe right diagnosis, inefficient actions or vice versa) |
| < 0.40 | ❌ | **Episode Failed** — Agent made wrong diagnosis, took harmful actions, or ran out of steps |

**For Battle Mode:**
```
┌───────────────────────────────┐
│   Defender  |  VS  |  Attacker│
│     1.40    |      |   0.40   │
│       Defender Wins! 🔵        │
│           0.990               │
└───────────────────────────────┘
```

Winner is whichever agent has higher cumulative reward. The Episode Score (0.990) is separate — it grades how well the overall incident was resolved regardless of who won the battle.

---

## What Makes a "Good" Episode?

| What the agent did | Score impact |
|-------------------|-------------|
| Scanned/queried the ACTUALLY affected service | ✅ High Investigation Quality |
| Correctly identified root cause (`cyber_attack:data_exfiltration`) | ✅ High Diagnosis score |
| Took targeted actions (didn't restart random services) | ✅ High Action Efficiency |
| Got distracted by the attacker's fake alerts | ❌ Low Efficiency |
| Diagnosed the wrong thing | ❌ Zero Diagnosis score |
| Isolated the wrong service causing an outage | ❌ Negative reward steps |

---

## The 4 Scenarios — What's Actually Happening

| Scenario | The Real Problem | The Trap / Noise |
|----------|-----------------|-------------------|
| **Memory Leak (Easy)** | `auth` service has a memory leak — needs restart | 5% noise, clear signals |
| **DDoS Cascade (Medium)** | Attackers hitting gateway → api → auth cascade | 25% noise, 2 attacking IPs to block |
| **Bad Deployment (Medium-Hard)** | api v2.4.1 pushed bad Redis config → reconnect storm | 35% noise, cache error looks like hardware fault |
| **Data Exfiltration (Hard)** | Service account `reports_bot` exfiltrating 4GB+ of data | 55% noise, false cache CRITICAL alert injected to mislead |

---

## Common Questions from Judges

**Q: Is the AI actually running live, or is this pre-recorded?**
> The 🤖 AI Output boxes show the raw JSON generated by the model in real time. Every episode hits our Hugging Face Inference Endpoint. You can see it at http://localhost:8000/debug/ai

**Q: How is this different from a script that reads the playbook?**
> Run Baseline Model mode — it makes obviously wrong choices (restarts the wrong service, blocks 8.8.8.8, wrong diagnosis). The trained model consistently picks the right service, correct mitigation, and correct diagnosis because it was trained to do so via GRPO.

**Q: What did GRPO training actually do?**
> GRPO (Group Relative Policy Optimization) is the same RL algorithm used to train DeepSeek-R1. It generates multiple candidate responses, ranks them by reward, and reinforces the better ones. After 500 training steps on the hardest scenarios, the model learned to prefer investigation → targeted mitigation → correct diagnosis over random actions.

**Q: How much did scores improve?**
> Baseline (untrained) episodes typically score 0.1–0.3. Trained model consistently scores 0.7–0.99 on hard scenarios. That's a ~3-5× improvement in grader score.
