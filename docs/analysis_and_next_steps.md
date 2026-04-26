# OpenSecOpsEnv — Full Audit, MDP Definition & Next Steps

---

## 1 · Full MDP Definition

This is the formal Markov Decision Process that your environment implements.

### State Space S

The **hidden state** (never given to the agent directly) is:

| Variable | Type | Description |
|---|---|---|
| `true_root_cause` | `str` | `"infra_failure"` or `"cyber_attack"` |
| `subtype` | `str` | `"memory_leak"`, `"ddos"`, `"data_exfiltration"` |
| `affected_services` | `list[str]` | Services that are actually compromised |
| `attack_progress` | `float ∈ [0,1]` | How far the attack/failure has progressed |
| `noise_level` | `float ∈ [0,1]` | Fraction of log/alert signals that are misleading |
| `metrics[svc]` | `dict` | Per-service `{cpu, memory, latency, error_rate}` |
| `isolated_services` | `set[str]` | Services the agent has isolated |
| `blocked_ips` | `set[str]` | IPs the agent has blocked |
| `step_count` | `int` | Current episode step |

The **observation** (what the agent actually sees) is a stochastic projection of the hidden state:

```
O = f(S) + noise
  = {alerts, metrics_snapshot, partial_logs, topology, last_action_result, time_step}
```

---

### Action Space A

Discrete (9 actions), parameterized:

| Action | Parameters | Class |
|---|---|---|
| `query_logs` | `{service}` | Investigation |
| `inspect_metrics` | `{service?}` | Investigation |
| `run_security_scan` | `{target}` | Investigation |
| `restart_service` | `{service}` | Mitigation |
| `scale_service` | `{service, replicas}` | Mitigation |
| `block_ip` | `{ip}` | Mitigation |
| `rollback_deployment` | `{service, version}` | Mitigation |
| `isolate_service` | `{service}` | Mitigation |
| `submit_diagnosis` | `{label}` | Terminal |

---

### Transition Function T(s, a) → s'

State evolves deterministically (seeded RNG) after each action:

1. **`_drift_metrics()`** — Brownian drift + scenario-specific escalation:
   - Memory leak: `memory += 1.5/step`, `latency += 10/step` per affected svc
   - DDoS: `cpu += attack_progress × 5`, `latency += attack_progress × 100`
   - Exfiltration: `cpu += attack_progress × 1.5`, `memory += attack_progress × 0.8`

2. **`_evolve_attack()`** — `attack_progress += 0.08/step` (×0.5 if an affected service is isolated)

3. **Mitigation side-effects** (when correct action taken):
   - `restart_service` → resets memory to 35%, latency to 80ms
   - `block_ip` → reduces affected svc cpu/latency, attack_progress −0.2
   - `isolate_service` → attack_progress −0.35
   - `scale_service` → cpu −20%, latency −200ms

---

### Reward Function R(s, a) → ℝ

Dense per-step reward ∈ [−1.0, +1.0]:

```
Investigation on affected service     → +0.2
inspect_metrics({}) (global)          → +0.2   (always useful)
run_security_scan on affected + cyber → +0.3
Correct mitigation                    → +0.5
Correct submit_diagnosis              → +1.0

Investigation on NON-affected svc     → −0.05
Ineffective mitigation                → −0.1
Wrong service (not in metrics)        → −0.2
Harmful action (wrong ip/isolate)     → −0.5
Wrong submit_diagnosis                → −1.0
```

---

### Terminal Condition

Episode ends when:
- Agent calls `submit_diagnosis` (always terminates, reward ±1.0), OR
- `step_count >= max_steps` (30 / 40 / 50 per task)

---

### Grader G(episode) → [0,1]

```
score = 0.5 × diagnosis_correct
      + 0.3 × action_efficiency
      + 0.2 × investigation_quality

diagnosis_correct  = 1.0 (exact) | 0.5 (category) | 0.0
action_efficiency  = 0.7 × mitigation_recall + 0.3 × step_bonus
  mitigation_recall = |done ∩ correct| / |correct|
  step_bonus        = min(1, ideal_steps / step_count)   ideal = max(n_correct×3, 5)
investigation_quality = |investigated_affected| / |affected|
```

---

## 2 · Gap Analysis vs Competition Rubric

### Real-world utility (30%) — **Current: ~25/30 → Target: 28/30**

| Issue | Impact | Fix? |
|---|---|---|
| Only 3 scenario types (memory_leak, DDoS, exfil) | Medium | Add 4th task type |
| False-alert signal in HARD task is labeled "FALSE POSITIVE" in the alert text itself | Medium | Remove the self-labeling hint |
| gateway not treated as a valid investigation target (penalized) | Low-Med | See §3-B |

---

### Task & grader quality (25%) — **Current: ~20/25 → Target: 24/25**

| Issue | Impact | Fix? |
|---|---|---|
| **run_security_scan counted as investigation not mitigation** — but HARD task lists it in `correct_mitigations`. Results in the agent never getting credit for scans even when done. | **Critical** | §3-A |
| Hard task: `affected_services = ["db","auth"]`, but agent gets no IQ credit for scanning — only for `query_logs`/`inspect_metrics`. `run_security_scan:db` should count toward investigation_quality | High | §3-A |
| grader `step_bonus` uses `ideal_steps = max(n_correct×3, 5)` — for easy task with 1 mitigation this gives `ideal=5` but optimal is 4 steps. Slightly penalizes perfect play | Low | §3-C |

---

### Environment design (20%) — **Current: ~17/20 → Target: 19/20**

| Issue | Impact | Fix? |
|---|---|---|
| `inspect_metrics({})` gives same +0.2 as targeted inspect. No penalty for already-seen services | Low | Cosmetic |
| `_generate_logs()` called twice per step (in `_build_observation` **and** in `_act_query_logs`) — same pool, different random shuffle | Low | §3-D |
| No cooldown on repeat actions — agent can `query_logs:auth` 20× for +4.0 reward with no diminishing returns | Medium | §3-E |

---

### Code quality & spec compliance (15%) — **Current: ~13/15 → Target: 15/15**

| Issue | Impact | Fix? |
|---|---|---|
| `score` in `log_end()` formatted as `:.2f` — spec says 2 decimal places, this is fine | None | — |
| `openenv.yaml` lists `run_security_scan` under `correct_mitigations` for HARD task, but grader never counts it as mitigation | Minor | §3-A |
| No `LICENSE` file despite README saying MIT | Low | §3-F |
| Docker HEALTHCHECK hits `/health` — ✓ good | — | — |
| `inference.py` falls back to heuristic when no LLM key — ✓ good | — | — |

---

### Creativity & novelty (10%) — **Current: ~8/10 → Target: 9/10**

| Issue | Impact | Fix? |
|---|---|---|
| The cache false-alert says "FALSE POSITIVE" in the message text — removes the discovery challenge | Medium | §3-B |
| Adding a 4th task would add novelty | Medium | §3-G (optional) |

---

## 3 · Prioritized Fixes (High → Low)

### A · **[CRITICAL] Fix run_security_scan in HARD grader** — `grader.py` + `env.py`

**Problem:** `run_security_scan:db` is listed in `correct_mitigations` for the hard task (`openenv.yaml` line 143-144), but `INVESTIGATION_ACTIONS` in `env.py` means it's never appended to `state.mitigation_actions` — only to `state.investigation_actions`. The grader only counts `mitigation_actions` toward `mitigation_recall`. So an agent doing the right scans gets 0 mitigation credit.

**Fix in `grader.py`:** Count security scans listed in `correct_mitigations` as valid — check both `mitigation_actions` and `investigation_actions` whose keys appear in `correct_mitigations`.

**Also fix in `grader.py`:** Let `run_security_scan:X` count toward `investigation_quality` for service X (already works because investigation_actions includes them, but the set intersection logic uses `parts[1]` which gives the target correctly — this part is actually fine).

```python
# In grade(), replace mitigation recall calculation:
# Check BOTH lists for items that appear in correct_mitigations
all_taken_actions = set(mitigation_actions) | set(investigation_actions)
n_correct = len(all_taken_actions & set(correct_mitigations))
```

---

### B · **[HIGH] Remove self-labeling hint from HARD task alert** — `task_definitions.py`

**Problem:** The false-alert message says `"(FALSE POSITIVE)"` in the text. This tells the agent exactly that it's a false alarm, defeating the point of the noise.

**Fix in `task_definitions.py`:** Change:
```python
# Before
"message": "Cache memory 82% – possible memory leak (FALSE POSITIVE)"
# After  
"message": "Cache memory 82% – triggering high-memory threshold policy"
```

---

### C · **[MEDIUM] Fix step_bonus for easy task** — `grader.py`

**Problem:** `ideal_steps = max(n_correct * 3, 5)` gives 5 for easy task (1 correct mitigation × 3 = 3, max with 5 = 5). But the optimal trajectory is 4 steps (inspect, query, restart, submit). An agent doing it perfectly in 4 steps gets `step_bonus = 5/4 = 1.25 → clamped to 1.0` — still OK. For medium: `3 × 3 = 9`, optimal is 7 — fine. This is actually acceptable, just add a note.

---

### D · **[MEDIUM] Prevent log double-generation and add repeat-action diminishing returns** — `env.py`

**Problem:** When `query_logs` is called, `_act_query_logs` calls `_generate_logs()` and then `_build_observation` also calls `_generate_logs()` again. Two different shuffles, slightly incoherent. Fix by building the observation first with the same log pool.

**Also:** An agent can harvest +0.2 per step by repeatedly calling `query_logs:auth` on the affected service. Add a per-service "already investigated" diminishing return.

**Fix in `env.py`:** Maintain a `_investigated: dict[str, int]` counter. First time: full reward. Second time same service: +0.05. Third+ time: 0.

```python
# In __init__:
self._investigated: dict[str, int] = {}

# In _act_query_logs / _act_inspect_metrics:
count = self._investigated.get(svc, 0)
self._investigated[svc] = count + 1
if svc in self._hidden.affected_services:
    reward = 0.2 if count == 0 else (0.05 if count == 1 else 0.0)
```

---

### E · **[LOW] Add LICENSE file** — root

Create a `LICENSE` file with the MIT license text. Required for open-source credibility and referenced by README.

---

### F · **[LOW / OPTIONAL] Add a 4th task** — `task_definitions.py` + `openenv.yaml`

A `medium_hard` task such as a **bad deployment rollback** scenario would:
- Fill the gap between medium (0.73) and hard (0.795) — scores are surprisingly close
- Add a `misconfiguration:bad_config` or `infra_failure:service_crash` scenario using `rollback_deployment` action (currently never the correct mitigation in any task!)

This is optional but would increase **creativity score** and make `rollback_deployment` useful.

---

## 4 · Files to Edit (Summary Table)

| File | Change | Priority |
|---|---|---|
| `opensecops_env/grader.py` | Fix mitigation_recall to include investigation_actions ∩ correct_mitigations | 🔴 Critical |
| `opensecops_env/tasks/task_definitions.py` | Remove "(FALSE POSITIVE)" from hard task alert text | 🟠 High |
| `opensecops_env/env.py` | Add diminishing returns for repeat investigation on same service | 🟡 Medium |
| `opensecops_env/env.py` | Fix double log generation (minor) | 🟡 Medium |
| `openenv.yaml` | Clarify that run_security_scan mitigations in HARD task are also graded as investigation | 🟡 Medium |
| `LICENSE` | Add MIT license text | 🟢 Low |
| `README.md` | Update baseline scores to actual measured values (1.00, 0.73, 0.795) | 🟢 Low |
| `opensecops_env/tasks/task_definitions.py` | (Optional) Add 4th task | 🔵 Optional |

---

## 5 · Should You Fine-tune?

**No, you do not need to fine-tune any model.** Your environment is the submission, not the model.

The competition evaluates **your environment quality** — judges will run their own models against it. The inference script produces a *baseline* score that shows the environment is solvable.

What you should optimize instead:

| Concern | What to do |
|---|---|
| Baseline score too low | Improve the system prompt in `inference.py` — more explicit strategy hints |
| Baseline score too high | Make tasks harder (more noise, more red herrings) |
| Scores not deterministic | Ensure `_rng` is seeded before EVERY `_generate_logs` call |
| Runtime > 20min | `MAX_STEPS = 20` already set — keep it; 3 tasks × 20 steps × ~1s/LLM call ≈ 1 min |

**System prompt improvements to `inference.py`** that would raise the baseline score:

1. Add explicit instruction: *"When investigating, always check ALL services in the topology, not just the most obvious one"*
2. Add: *"The gateway / frontend service is rarely the root cause; look at the backend services it calls"*
3. Add: *"You MUST call `run_security_scan` on suspicious services before submitting if you suspect a cyber attack"*

---

## 6 · Deployment Checklist (Before Submitting to HF Spaces)

- [ ] Run `docker build -t opensecops-env:latest .` locally ✓ (user is doing this now)
- [ ] Run `docker run -p 8000:8000 opensecops-env:latest` and verify `/health` returns 200
- [ ] `curl -X POST http://localhost:8000/reset -d '{}' -H 'Content-Type: application/json'` returns observation
- [ ] Set `HF_TOKEN` and run `python inference.py` — verify 3× `[START]`/`[STEP]`/`[END]` blocks print
- [ ] Push to HF Spaces with correct `README.md` `sdk: docker` header
- [ ] Set Space secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Ping `https://your-space.hf.space/reset` with POST — verify 200

### HF Spaces `README.md` header (required at top of HF repo README):

```yaml
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
  - agent-evaluation
---
```
