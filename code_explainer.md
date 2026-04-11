# OpenSecOpsEnv — Deep Technical Explainer

---

## Section 1: Did You Make an RL Agent? (The Most Important Thing to Understand)

### Short Answer
**No — you made the ENVIRONMENT. That is what's being graded.**

### Long Answer

In Reinforcement Learning (RL), there are always two distinct pieces:

```
┌─────────────────────┐         action         ┌─────────────────────┐
│                     │ ───────────────────────▶│                     │
│       AGENT         │                         │    ENVIRONMENT      │
│  (the decision-     │◀─────────────────────── │  (the simulation    │
│   maker / policy)   │   observation + reward  │   of the world)     │
└─────────────────────┘                         └─────────────────────┘
```

- The **ENVIRONMENT** is the simulator. It defines: what the world looks like, what actions are possible, how actions change the world, and what reward the agent gets. Like a chess engine, or an Atari game emulator.
- The **AGENT** is the thing that learns to play. It observes the world and picks actions.

**You built the ENVIRONMENT** — `OpenSecOpsEnv`. The entire `opensecops_env/` package is your submission.

### What is `inference.py` Then?

`inference.py` contains a **baseline agent** — required by the competition rules to prove your environment works and produces sensible scores. It's NOT what's being judged.

```
What you're judged on:  The quality of the ENVIRONMENT
What inference.py is:   A demonstration that the environment works
```

### Is the LLM an RL Agent?

**Not a traditional RL agent — it's a prompted LLM acting as a policy.** Here's the distinction:

| | Traditional RL Agent | Your LLM Agent |
|---|---|---|
| Learning | Trains by trial and error for millions of steps | No training — uses pre-trained weights |
| Memory | Updates its weights based on rewards | No weight updates — just a conversation history |
| Exploration | Tries random exploratory actions early | Relies on the system prompt for strategy |
| How it gets better | Gradient descent over many episodes | Prompt engineering |

The LLM "acts" like a policy — it takes observations in, outputs actions — but it doesn't *learn* from rewards. Future RL researchers would use your environment to *actually* train RL agents (PPO, RLHF, etc.) after the competition.

### The Competition Flow

```
1. You submit: environment + inference.py + Dockerfile + README
2. Judges validate: docker build, openenv validate, 3+ tasks, outputs [START]/[STEP]/[END]
3. Judges run THEIR own agent (Nemotron, Llama, etc.) against YOUR environment
4. Judges grade YOUR ENVIRONMENT on: how real is the task? How good are the graders?
   How meaningful is the reward signal? Are the tasks well-designed?
5. You win if your environment is the best benchmark for training/evaluating agents
```

---

## Section 2: Complete File-by-File Code Explanation

---

### FILE: `opensecops_env/models.py` — The Data Contracts

This file defines the "language" everything else speaks. Every piece of data passed between the agent, environment, and grader is typed here.

```python
class ActionType(str, Enum):
    QUERY_LOGS = "query_logs"
    INSPECT_METRICS = "inspect_metrics"
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    BLOCK_IP = "block_ip"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    RUN_SECURITY_SCAN = "run_security_scan"
    ISOLATE_SERVICE = "isolate_service"
    SUBMIT_DIAGNOSIS = "submit_diagnosis"
```
> **Why an Enum?** Makes it impossible to accidentally pass a typo like `"query_log"`. The env dispatches on these exact strings.

```python
@dataclass
class SecOpsAction:
    action_type: str        # one of the 9 ActionType values
    parameters: dict        # action-specific args, e.g. {"service": "auth"}
```
> **This is what the agent sends.** One action per step. The env receives this.

```python
@dataclass
class ServiceMetrics:
    cpu: float          # 0-100 %
    memory: float       # 0-100 %
    latency: float      # milliseconds (p99)
    error_rate: float   # 0-100 %
```
> **Per-service health numbers.** The environment mutates these every step to simulate time passing.

```python
@dataclass
class SecOpsObservation:
    alerts: list[dict]             # threshold alerts, e.g. "auth CPU 95%"
    metrics: dict[str, dict]       # all services' {cpu, memory, latency, error_rate}
    logs: list[str]                # up to 8 recent log lines (noisy, partial)
    topology: dict[str, list]      # service dep graph: {"api": ["auth", "db"]}
    last_action_result: str        # text feedback from the previous action
    time_step: int                 # current step count
    available_actions: list[str]   # constant list of all 9 action types
```
> **This is what the agent SEES.** Critically, it does NOT contain the true root cause.
> The agent must *infer* the root cause from these noisy clues.

```python
@dataclass
class HiddenState:
    true_root_cause: str       # "infra_failure" or "cyber_attack" or "misconfiguration"
    subtype: str               # "memory_leak", "ddos", "data_exfiltration", "bad_config"
    affected_services: list    # which services are ACTUALLY broken
    attack_progress: float     # 0-1, how far the attack has spread
    noise_level: float         # what % of signals are misleading
    diagnosis_submitted: bool  # did the agent submit yet?
    submitted_label: str       # what label did they submit?
```
> **This is what the environment KNOWS but the agent CANNOT SEE directly.**
> This is the "ground truth" that drives all reward calculations.
> The agent must deduce this through investigation.

```python
@dataclass
class SecOpsState:
    episode_id: str                  # unique ID for this episode
    step_count: int                  # how many steps taken
    task_id: str                     # which task is running
    max_steps: int                   # episode length limit
    cumulative_reward: float         # running total of all rewards
    done: bool                       # is the episode over?
    investigation_actions: list[str] # e.g. ["query_logs:auth", "inspect_metrics:all"]
    mitigation_actions: list[str]    # e.g. ["restart_service:auth", "block_ip:1.2.3.4"]
    correct_mitigations: list[str]   # the gold standard answer set
    hidden_state: dict               # the HiddenState as a plain dict
```
> **This is exposed by the `state()` endpoint.** It contains the hidden state too
> (for graders and debuggers). The agent is NOT meant to read this directly.

---

### FILE: `opensecops_env/tasks/task_definitions.py` — The 4 Scenarios

This file defines all 4 tasks as Python dicts. It's the "game data" — like a game designer's level definitions.

Each task is created by calling `_task(...)` which is just a dict constructor with named fields:

```python
TASK_EASY = _task(
    task_id="easy_memory_leak",
    difficulty="easy",
    seed=42,                        # RNG seed — guarantees identical episodes every time
    true_root_cause="infra_failure",
    subtype="memory_leak",
    affected_services=["auth"],     # ONLY auth is broken
    noise_level=0.05,               # only 5% misleading signals
    max_steps=30,
    correct_mitigations=["restart_service:auth"],
    correct_label="infra_failure:memory_leak",
    initial_metrics={
        "auth": {"cpu": 45.0, "memory": 88.0, "latency": 320.0, "error_rate": 2.1},
        # auth memory=88% is a BIG clue — close to OOM
        "api":  {"cpu": 22.0, "memory": 35.0, "latency": 80.0, "error_rate": 0.3},
        "db":   {"cpu": 18.0, "memory": 55.0, "latency": 12.0, "error_rate": 0.1},
    },
    initial_logs=[
        "[auth] WARN  OutOfMemoryError imminent – heap 88% full",
        "[auth] ERROR Request timeout: downstream DB query exceeded 300ms",
        # clear signals pointing at auth
    ],
    ...
)
```

**Key design decisions per task:**

| Task | Difficulty signal | How you get tricked |
|---|---|---|
| easy_memory_leak | Clear single-service signals | Nothing misleading — learn the pattern |
| medium_ddos_cascade | Gateway shows worst metrics but ISN'T the root cause | Agent focuses on gateway (most alarming) not api/auth (the real victims) |
| medium_hard_bad_deployment | Bad api deploy → cache breaks via bad Redis config | Symptoms look like cache failure, not an api configuration issue |
| hard_data_exfiltration | DB exfil disguised as performance issue + 55% noise + false critical/alert on cache | False positive cache alert, security scan required to discover the true threat |

```python
TASKS: dict[str, TaskConfig] = {
    "easy_memory_leak":          TASK_EASY,
    "medium_ddos_cascade":       TASK_MEDIUM,
    "medium_hard_bad_deployment": TASK_MEDIUM_HARD,
    "hard_data_exfiltration":    TASK_HARD,
}

def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASKS)}")
    return TASKS[task_id]
```
> `get_task()` is called by `env.reset()` to load the scenario.

---

### FILE: `opensecops_env/env.py` — The Simulation Engine (Most Important File)

This is the heart of the project. ~730 lines. Let's go through every section.

#### Module-level constants

```python
INVESTIGATION_ACTIONS = {
    ActionType.QUERY_LOGS.value,
    ActionType.INSPECT_METRICS.value,
    ActionType.RUN_SECURITY_SCAN.value,
}

MITIGATION_ACTIONS = {
    ActionType.RESTART_SERVICE.value,
    ActionType.SCALE_SERVICE.value,
    ActionType.BLOCK_IP.value,
    ActionType.ROLLBACK_DEPLOYMENT.value,
    ActionType.ISOLATE_SERVICE.value,
}
```
> Used to classify actions for tracking `investigation_actions` vs `mitigation_actions` lists in state.

---

#### `_drift_metrics()` — Time Passing

```python
def _drift_metrics(metrics, hidden, rng, noise_level):
    for svc, m in metrics.items():
        # BROWNIAN MOTION: every service randomly wobbles slightly each step
        m.cpu       += rng.gauss(0, noise_level * 2)   # gauss = normal distribution
        m.memory    += rng.gauss(0, noise_level * 1.5)
        m.latency   += rng.gauss(0, noise_level * 5)
        m.error_rate = max(0.0, m.error_rate + rng.gauss(0, noise_level * 0.5))

        # SCENARIO-SPECIFIC ESCALATION:
        if rc == "infra_failure" and sub == "memory_leak":
            if svc in hidden.affected_services:   # only for auth
                m.memory += 1.5       # memory keeps climbing — the leak isn't fixed
                m.latency += 10.0     # GC pressure slows responses
                m.error_rate += 0.3   # timeouts start accumulating

        elif rc == "cyber_attack" and sub == "ddos":
            if svc in hidden.affected_services:
                spike = ap * 5.0       # ap = attack_progress (0-1)
                # As attack progresses, affected services get hammered harder
                m.cpu += spike + rng.gauss(0, 2)
                m.latency += spike * 20
                m.error_rate += spike * 2

        # All values clamped to valid ranges after mutation
        m.cpu = max(0.0, min(100.0, m.cpu))
        ...
```

**Why this matters for the competition judges:**
This is what makes the environment feel *alive*. Without drift, the agent could wait indefinitely and nothing would change. With drift, delay = rising risk. The DDoS keeps getting worse. The memory leak progresses. This creates *temporal pressure* — a key property of real-world incident response.

---

#### `_generate_logs()` — The Log Stream

```python
def _generate_logs(hidden, rng, task_initial_logs, step):
    pool = list(task_initial_logs)   # start with static logs from task config

    # Add DYNAMIC logs based on current scenario state:
    if rc == "infra_failure" and sub == "memory_leak":
        pool += [
            f"[auth] WARN  Heap dump triggered (step {step}): 89% used",
            "[auth] ERROR Java heap space – GC overhead limit exceeded",
        ]

    elif rc == "cyber_attack" and sub == "ddos":
        if ap > 0.5:   # only after attack has progressed far enough
            pool += [
                "[gateway] CRIT  SYN flood detected from /24 block 203.0.113.0",
                "[api]  ERROR  Worker pool exhausted – dropping requests",
            ]

    # INJECT NOISE: add innocent-looking logs
    noise_logs = [
        "[syslog] INFO  NTP sync OK",
        "[kernel] INFO  Disk I/O stable",
        "[cron]   INFO  Backup job completed successfully",
        ...
    ]
    n_noise = int(len(pool) * hidden.noise_level)  # 5% noise = 1 noisy log, 55% = many
    pool += rng.sample(noise_logs, min(n_noise, len(noise_logs)))

    rng.shuffle(pool)   # randomise order (makes it feel like a real log stream)
    return pool[:8]     # agent only sees 8 most recent lines — partial window
```

**Key insight:** The agent never gets ALL the logs, only a sliding window of 8. This forces the agent to actively `query_logs` on specific services to get targeted information, rather than passively reading everything.

---

#### `_evolve_attack()` — The Ticking Clock

```python
def _evolve_attack(hidden, isolated):
    if hidden.true_root_cause != "cyber_attack":
        return   # irrelevant for infra failures

    # If agent isolated an affected service, attack spreads at half speed
    slowdown = 0.5 if any(s in isolated for s in hidden.affected_services) else 1.0
    hidden.attack_progress = min(1.0, hidden.attack_progress + 0.08 * slowdown)
```

> `attack_progress` starts at 0.3 for DDoS, 0.55 for data exfil. Each step it grows by 0.08 (unless contained). As it grows, the DDoS becomes more severe (more cpu/latency spike in `_drift_metrics`). Isolation slows it — a direct incentive to act quickly.

---

#### `__init__()` — Where All State Lives

```python
def __init__(self):
    self._state = SecOpsState()          # episode metadata (step count, rewards, action history)
    self._hidden = HiddenState()          # ground truth (agent can't see this)
    self._metrics = {}                    # mutable per-service metrics
    self._rng = random.Random(42)         # seeded RNG for reproducibility
    self._task_cfg = {}                   # loaded task definition

    # Tracking sets (what the agent has done this episode)
    self._isolated_services = set()
    self._blocked_ips = set()
    self._restarted_services = set()
    self._scaled_services = set()
    self._rolled_back = set()
    self._scanned = set()

    # Diminishing returns counter
    self._investigated = {}   # {"auth": 2, "db": 1} — how many times queried
```

---

#### `reset(task_id)` — Start of Episode

```python
def reset(self, task_id="easy_memory_leak"):
    cfg = get_task(task_id)         # load the task dict from task_definitions.py
    self._task_cfg = cfg

    # REPRODUCIBILITY: same seed → identical episode every time
    self._rng = random.Random(cfg["seed"])

    # CLEAN SLATE: wipe all tracking state
    self._isolated_services = set()
    self._blocked_ips = set()
    # ... all tracking sets reset ...
    self._investigated = {}

    # BUILD HIDDEN STATE (ground truth the agent can't see)
    self._hidden = HiddenState(
        true_root_cause=cfg["true_root_cause"],     # "cyber_attack"
        subtype=cfg["subtype"],                      # "ddos"
        affected_services=list(cfg["affected_services"]),  # ["api", "auth"]
        attack_progress=cfg.get("attack_progress_start", 0.0),
        noise_level=cfg["noise_level"],
    )

    # BUILD LIVE METRICS from task config's initial values
    self._metrics = {
        svc: ServiceMetrics(**vals)         # ServiceMetrics(cpu=91.0, memory=62.0, ...)
        for svc, vals in cfg["initial_metrics"].items()
    }

    # BUILD EPISODE STATE (tracking object)
    self._state = SecOpsState(
        episode_id=str(uuid.uuid4()),       # unique ID for this run
        step_count=0,
        task_id=task_id,
        max_steps=cfg["max_steps"],
        cumulative_reward=0.0,
        done=False,
        correct_mitigations=list(cfg["correct_mitigations"]),
    )

    # RETURN FIRST OBSERVATION (what agent sees at start)
    return self._build_observation("Episode started. Investigate the system.")
```

---

#### `step(action)` — One Step of the Loop

```python
def step(self, action):
    # Guard: don't process actions in a finished episode
    if self._state.done:
        return self._build_observation("Episode already finished."), 0.0, True, {}

    self._state.step_count += 1

    # CORE: dispatch to the right action handler, get reward + message
    reward, result_msg, done = self._process_action(action)

    # DYNAMICS: advance time (metrics drift, attack grows)
    _drift_metrics(self._metrics, self._hidden, self._rng, self._hidden.noise_level)
    _evolve_attack(self._hidden, self._isolated_services)

    # TERMINATION CHECK: did we hit the step limit?
    if self._state.step_count >= self._state.max_steps:
        done = True
        result_msg += " [Max steps reached]"

    # BOOKKEEPING
    self._state.done = done
    self._state.cumulative_reward += reward

    # BUILD NEXT OBSERVATION (what agent sees after this action)
    obs = self._build_observation(result_msg)

    return obs, round(reward, 4), done, {"step": self._state.step_count, ...}
```

**The crucial point:** `_process_action` runs BEFORE `_drift_metrics`. So the agent's action (e.g., blocking an IP) takes effect first, THEN time passes and metrics drift. This is physically correct — you block the IP, then the system stabilises.

---

#### Action Handlers — The Core Logic

Each action handler returns `(reward: float, message: str, done: bool)`.

**`_act_query_logs(params)`**
```python
svc = params.get("service", "")
if svc not in self._metrics:
    return -0.2, f"Service '{svc}' not found.", False  # unknown service

# RECORD: this service was investigated
self._state.investigation_actions.append(f"query_logs:{svc}")

# GENERATE LOG CONTENT: filter the full pool to only lines mentioning svc
logs = _generate_logs(self._hidden, self._rng, ...)
relevant = [l for l in logs if f"[{svc}]" in l or svc in l.lower()]

# REWARD: diminishing returns
if svc in self._hidden.affected_services:
    times = self._investigated.get(svc, 0)
    self._investigated[svc] = times + 1
    reward = 0.2 if times == 0 else (0.05 if times == 1 else 0.0)
    # First look = full reward. Second = tiny bonus. Third+ = nothing.
else:
    reward = -0.05  # mild penalty for wasting time on healthy service
```

**`_act_restart_service(params)`**
```python
key = f"restart_service:{svc}"

if key in self._task_cfg.get("correct_mitigations", []):
    # CORRECT MITIGATION: reset this service's metrics
    self._metrics[svc].memory = 35.0      # memory leak cleared
    self._metrics[svc].latency = 80.0     # latency normalised
    self._metrics[svc].error_rate = 0.5   # errors settled
    self._state.mitigation_actions.append(key)
    reward = 0.5
    msg = f"Service '{svc}' restarted successfully."

elif svc not in self._hidden.affected_services:
    # HARMFUL: restarting a healthy service causes a brief outage
    m.error_rate = min(100.0, m.error_rate + 5.0)   # users experience errors!
    m.latency = min(5000.0, m.latency + 200.0)
    reward = -0.5   # heavy penalty
    msg = f"Service '{svc}' restarted unnecessarily. Caused brief outage."

else:
    # Affected service, wrong action (e.g. restarting during DDoS doesn't help)
    reward = -0.2
    msg = f"Restarting '{svc}' had minimal effect — root cause needs different fix."
```

**`_act_block_ip(params)`**
```python
key = f"block_ip:{ip}"
correct_blocks = [m for m in correct_mitigations if m.startswith("block_ip:")]

if key in correct_blocks:
    # CORRECT: reduce attack traffic → metrics improve
    for svc in self._hidden.affected_services:
        self._metrics[svc].cpu -= 15.0
        self._metrics[svc].latency -= 200.0
    self._hidden.attack_progress -= 0.2    # attack retreats
    reward = 0.5

else:
    # BLOCKING LEGIT IP: every service starts getting errors
    for svc in self._metrics.values():
        svc.error_rate = min(100.0, svc.error_rate + 2.0)  # widespread damage
    reward = -0.5
```

> **Why the harsh penalty?** Blocking legitimate IPs (like CDNs, payment gateways) causes customer-facing outages. This mirrors real-world consequences and discourages the agent from randomly blocking IPs it found in logs.

**`_act_run_security_scan(params)`**
```python
if rc == "cyber_attack" and target in self._hidden.affected_services:
    # Target IS compromised AND this IS a cyber attack
    if sub == "data_exfiltration":
        msg = "SECURITY SCAN – {target}: ALERT: Unusual outbound data transfer detected.
               Suspicious process 'reports_bot' opened 847 connections. Recommend isolation."
    elif sub == "ddos":
        msg = "SECURITY SCAN – {target}: ALERT: Connection table 98% full –
               SYN flood indicators. Recommend blocking attacking IP ranges."

    # Diminishing returns: +0.3 first scan, 0 on repeat
    already = f"run_security_scan:{target}" in self._state.investigation_actions[:-1]
    reward = 0.0 if already else 0.3
else:
    msg = f"SECURITY SCAN – {target}: No critical vulnerabilities found."
    reward = -0.05
```

> **This is the "eureka" moment for cyber attack tasks.** The first security scan on an affected service reveals the smoking gun.

**`_act_submit_diagnosis(params)`**
```python
label = params.get("label", "")
correct = self._task_cfg.get("correct_label", "")

if label == correct:
    reward = 1.0
    msg = f"✅ Correct diagnosis: '{label}'. Episode complete."
else:
    reward = -1.0
    msg = f"❌ Incorrect diagnosis: '{label}'. True root cause: '{correct}'."

return reward, msg, True   # always True — submitting ALWAYS ends the episode
```

> **Always terminates.** The agent can't "take it back". This is a real-world constraint — once you escalate an incident and submit your report, you're committed.

---

#### `_build_observation()` — Constructing What the Agent Sees

```python
def _build_observation(self, last_action_result):
    # Snapshot current metrics (as plain dicts, not ServiceMetrics objects)
    metrics_snapshot = {
        svc: {"cpu": round(m.cpu, 2), "memory": round(m.memory, 2), ...}
        for svc, m in self._metrics.items()
    }

    # Generate dynamic alerts from current metric values
    alerts = self._generate_alerts()

    # Generate a log slice (shuffled, noisy, partial window of 8)
    logs = _generate_logs(self._hidden, self._rng, ...)

    return SecOpsObservation(
        alerts=alerts,
        metrics=metrics_snapshot,
        logs=logs,
        topology=self._task_cfg.get("topology", {}),
        last_action_result=last_action_result,  # text of what just happened
        time_step=self._state.step_count,
    )
```

**`_generate_alerts()`** — threshold scanner:
```python
for svc, m in self._metrics.items():
    if m.cpu > 90:     alerts.append({service, type:"high_cpu", severity:"critical"})
    elif m.cpu > 75:   alerts.append({service, type:"high_cpu", severity:"warning"})
    if m.memory > 85:  alerts.append({service, type:"high_memory", severity:"critical"})
    ...

# Also inject static misleading alerts from task config
for a in self._task_cfg.get("initial_alerts", []):
    if self._rng.random() < noise:   # inject probabilistically
        alerts.append(a)
```

---

### FILE: `opensecops_env/grader.py` — The Judge

Called after every episode ends. Takes the episode state dict, returns a `GradeResult(score=0.xxx)`.

```python
def grade(episode_state: dict) -> GradeResult:

    # Pull everything out of the state dict
    hidden = episode_state["hidden_state"]
    correct_label = hidden["correct_label"]         # "cyber_attack:ddos"
    submitted_label = hidden["submitted_label"]      # what the agent submitted

    investigation_actions = episode_state["investigation_actions"]
    # e.g. ["inspect_metrics:all", "query_logs:api", "run_security_scan:api"]
    mitigation_actions = episode_state["mitigation_actions"]
    # e.g. ["block_ip:203.0.113.45", "block_ip:198.51.100.12", "scale_service:api"]
    correct_mitigations = episode_state["correct_mitigations"]
    affected_services = set(hidden["affected_services"])
```

#### Component 1: Diagnosis Correctness

```python
diagnosis_correct = 1.0 if submitted_label == correct_label else 0.0

# Partial credit: if they got the category right but wrong subtype
# e.g. submitted "infra_failure:service_crash" when correct is "infra_failure:memory_leak"
if diagnosis_correct == 0.0 and correct_label:
    correct_category = correct_label.split(":")[0]        # "infra_failure"
    submitted_category = submitted_label.split(":")[0]    # "infra_failure"
    if correct_category == submitted_category:
        diagnosis_correct = 0.5   # half credit for getting the class right
```

#### Component 2: Action Efficiency

```python
# Critical fix: check BOTH lists because run_security_scan goes to investigation_actions
# but some tasks list scans in correct_mitigations
all_taken = set(mitigation_actions) | set(investigation_actions)
n_correct = len(all_taken & set(correct_mitigations))

mitigation_recall = n_correct / max(n_total_correct, 1)
# e.g. did 3/4 required actions → 0.75

# Bonus for being fast
ideal_steps = max(n_total_correct * 3, 5)   # rough budget: 3 steps per mitigation
step_ratio = ideal_steps / max(step_count, 1)
step_bonus = min(1.0, step_ratio)            # capped at 1.0

action_efficiency = 0.7 * mitigation_recall + 0.3 * step_bonus
```

#### Component 3: Investigation Quality

```python
investigated = set()
for act in investigation_actions:
    # "query_logs:auth" → svc = "auth"
    # "run_security_scan:db" → svc = "db"
    parts = act.split(":", 1)
    if len(parts) == 2:
        svc = parts[1]
        if svc in affected_services:
            investigated.add(svc)

investigation_quality = len(investigated) / len(affected_services)
# e.g. looked at both "db" and "auth" → 2/2 = 1.0
# looked at only "db" → 1/2 = 0.5
```

#### Final Score

```python
score = (
    0.5 * diagnosis_correct        # 50% — must get the label right
    + 0.3 * action_efficiency      # 30% — did you actually fix it?
    + 0.2 * investigation_quality  # 20% — did you look at all affected services?
)
score = round(max(0.0, min(1.0, score)), 4)   # clamp to [0, 1]
```

---

### FILE: `opensecops_env/server/app.py` — The HTTP API

Wraps the environment in a FastAPI web server. This is what runs inside Docker and what HF Spaces exposes to the public internet.

**Session model (improved):**
```python
_sessions: dict[str, OpenSecOpsEnv] = {}  # session_id → env instance
_default_session: str = "default"

def _get_session(session_id):
    if session_id not in _sessions:
        raise HTTPException(400, "Session not found. Call /reset first.")
    return _sessions[session_id]
```
> Each `/reset` call creates a FRESH `OpenSecOpsEnv()` instance. Two agents running `medium_ddos` and `hard_exfil` simultaneously get two separate env instances with no shared state.

**`POST /reset`**
```json
Request:  {"task_id": "easy_memory_leak", "session_id": ""}
Response: {"session_id": "easy_memory_leak", "observation": {...}}
```
Creates a new env, seeds it, returns first observation.

**`POST /step`**
```json
Request:  {"action_type": "query_logs", "parameters": {"service": "auth"}, "session_id": "easy_memory_leak"}
Response: {"observation": {...}, "reward": 0.20, "done": false, "info": {"step": 1}}
```

**`GET /state`** — Debug endpoint. Returns the full internal state including hidden state. Used by graders, not intended for agents.

**`GET /tasks`** — Returns metadata for all 4 tasks. Added so the competition's automated tooling can discover tasks programmatically.

---

### FILE: `inference.py` — The Baseline Agent

```python
# CONFIGURATION — read from environment variables (required by competition rules)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# Build the OpenAI-compatible client (works with HF inference router)
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
```

**The Agent Loop (one task):**
```python
def run_task(client, task_id):
    env = OpenSecOpsEnv()
    obs = env.reset(task_id)          # fresh episode
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task_id, env="opensecops", model=MODEL_NAME)

    for step_n in range(1, MAX_STEPS + 1):
        # Format observation as readable text for the LLM
        obs_text = _obs_to_text(obs_dict)
        messages.append({"role": "user", "content": obs_text})

        # Ask the LLM what action to take
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,        # low temp = more deterministic
            max_tokens=300,
        )
        raw_action = response.choices[0].message.content

        # Parse the JSON response into a SecOpsAction
        action = _parse_action(raw_action)

        # Send action to environment, get back result
        obs, reward, done, info = env.step(action)

        # MANDATORY: print the [STEP] log line
        log_step(step=step_n, action=str(action), reward=reward, done=done, error=None)

        if done:
            break

    # Grade the completed episode
    grade_result = grade(env.state.to_dict())
    score = grade_result.score

    # MANDATORY: print the [END] line
    log_end(success=(score >= 0.5), steps=steps_taken, score=score, rewards=rewards)
```

**The `_obs_to_text()` function** converts the structured observation into readable prose the LLM can reason about:
```
=== Step 3 ===
Last action result: Logs for 'auth': [auth] WARN OutOfMemoryError imminent...

ALERTS:
  [CRITICAL] auth – high_memory: auth memory 90.1%

METRICS:
  auth: cpu=46.2% mem=90.1% lat=340ms err=2.4%
  api:  cpu=22.5% mem=35.1% lat=81ms err=0.3%

LOGS (recent, may contain noise):
  [auth] ERROR Java heap space – GC overhead limit exceeded
  [syslog] INFO NTP sync OK
  ...

TOPOLOGY (service → dependencies):
  api → ['auth', 'db']
  auth → ['db']
```

**The System Prompt** (what makes the LLM a good agent):
```
You are an on-call SRE/security engineer.
1. ALWAYS start with inspect_metrics({}) to see ALL services at once.
2. Gateway/proxy services propagate problems from backend services.
   Focus on backend (api, auth, db, cache), not the gateway.
3. Investigate ALL services in affected topology, not just the worst-looking one.
4. If logs mention IPs or security events, run run_security_scan BEFORE mitigating.
5. If logs mention a recent deployment, check rollback_deployment.
6. Never block an IP unless you saw it in logs as a threat source.
...
```

**The Heuristic Fallback** — when no API key:
```python
_HEURISTIC = {
    "easy_memory_leak": [
        {"action_type": "inspect_metrics", "parameters": {}},
        {"action_type": "query_logs", "parameters": {"service": "auth"}},
        {"action_type": "restart_service", "parameters": {"service": "auth"}},
        {"action_type": "submit_diagnosis", "parameters": {"label": "infra_failure:memory_leak"}},
    ],
    # ... etc for all 4 tasks
}
```

---

### FILE: `openenv.yaml` — The Contract Declaration

This file is what `openenv validate` parses. It must match the actual implementation exactly. Key sections:

```yaml
spec_version: 1
name: opensecops
type: container
runtime: docker
app: opensecops_env.server.app:app
port: 8000

observation:
  type: object
  properties:
    alerts:       { type: array }
    metrics:      { type: object }
    logs:         { type: array, items: {type: string} }
    topology:     { type: object }
    last_action_result: { type: string }
    time_step:    { type: integer }
    available_actions: { type: array }

action:
  type: object
  required: [action_type]
  properties:
    action_type:
      type: string
      enum: [query_logs, inspect_metrics, restart_service, scale_service,
             block_ip, rollback_deployment, run_security_scan,
             isolate_service, submit_diagnosis]
    parameters:
      type: object

tasks:
  - id: easy_memory_leak
    difficulty: easy
    seed: 42
    ...
  - id: medium_ddos_cascade
    ...
  - id: medium_hard_bad_deployment
    ...
  - id: hard_data_exfiltration
    ...

grader:
  score_range: [0.0, 1.0]
  formula: "0.5 * diagnosis_correct + 0.3 * action_efficiency + 0.2 * investigation_quality"
```

---

### FILE: `tests/test_opensecops.py` — 33 Unit Tests

Organised in 5 test classes:

| Class | What It Tests | Count |
|---|---|---|
| `TestTaskDefinitions` | All 4 tasks registered, required fields present, invalid task raises error | 3 |
| `TestReset` | Returns observation, initialises clean state, reproducible, hides root cause | 5 |
| `TestStep` | All action reward signs, termination conditions, post-episode behavior | 12 |
| `TestState` | Hidden state exposed in state(), to_dict() has all fields | 2 |
| `TestGrader` | Output range, all 4 tasks score ≥ 0.8 optimally, wrong=low, partial credit, deterministic | 8 |
| `TestEndToEnd` | Full episode runs without error, metrics change over time | 2 |

Every test is designed so if you break something in `env.py` or `grader.py`, at least one test will catch it.

---

## Section 3: Reward Design — Why It's Good (For Judges)

The competition gives points for "meaningful reward signal". Here's why yours is good:

### It's Dense (not sparse)

- **Sparse reward** = only +1 at the very end if you won, 0 all other steps. Horrible for learning.
- **Your reward** = signal at EVERY step. A positive signal (+0.2) tells the agent "looking at auth logs after seeing the memory alert was a good move." A negative signal (-0.05) tells it "you wasted a step on a healthy service."

### It Has Natural Structure

```
+0.20  correct investigation  (you gathered useful evidence)
+0.30  correct inference      (security scan proved it's an attack)
+0.50  correct mitigation     (you actually fixed part of the problem)
+1.00  correct diagnosis      (you named the root cause correctly)

-0.05  irrelevant investigation (mild: you just wasted a step)
-0.10  ineffective mitigation   (medium: wrong tool for the job)
-0.50  harmful action           (serious: you made things worse for users)
-1.00  wrong diagnosis          (worst: escalated with wrong information)
```

The rewards are proportional to real-world consequences. Blocking the wrong IP is 10x worse than checking the wrong service's logs, because one causes an outage and the other just wastes a minute.

### Diminishing Returns Prevent Reward Hacking

Without diminishing returns, the optimal "agent" would be:
```
query_logs:auth × 20 steps → +4.0 cumulative reward without doing any work
submit_diagnosis → +1.0
Total: +5.0
```

With diminishing returns:
```
query_logs:auth → +0.20  (first time, useful)
query_logs:auth → +0.05  (second time, maybe useful)
query_logs:auth → +0.00  (third time, you already know)
```

This forces the agent to explore broadly rather than exploit one action repeatedly.

---

## Section 4: Questions a Judge Might Ask

**Q: Why SecOps specifically?**
A: It's a $1.5T market problem. Real companies lose millions per hour during production incidents. An agent that can handle even routine alerts would be directly valuable. The domain hasn't been benchmarked in OpenEnv before.

**Q: How is the environment reproducible?**
A: Every task has a fixed seed. `self._rng = random.Random(cfg["seed"])`. The same seed + same action sequence = identical episode every time. The task configs are pure data (no randomness in task definitions). This is required for fair comparison across different agents.

**Q: What's the partial observability design?**
A: The agent sees `SecOpsObservation`. It does NOT see `HiddenState`. The true root cause, affected services, and attack progress are invisible. The agent must infer them from noisy partial signals — alerts that threshold-fire based on metrics, log lines filtered to 8, and the topology graph.

**Q: What prevents a trivial agent from scoring 1.0 on every task?**
A: The hard task has 55% noise level, false alerts on innocent services, and requires running security scans (not just logs) to discover the attack. A random agent would score ~0.15 (0.5 × 0 diagnosis + 0.3 × small + 0.2 × small). Even frontier LLMs like Qwen 72B score ~0.80 on hard.

**Q: How do concurrent requests work?**
A: Session-based. Each `POST /reset` creates a fresh `OpenSecOpsEnv()` instance stored in a dict keyed by `session_id`. Multiple agents can run simultaneously without sharing state.
