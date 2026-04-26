# OpenSecOpsEnv — Complete Code Reference

> **Auto-generated reference for the current codebase. Last updated: April 2026.**

---

## Project Structure

```
incident-ai/
├── README.md                        # Main submission README (judges start here)
├── hf_blog_post.md                  # HF blog post (copy to model card)
├── colab_training (2).ipynb         # GRPO training notebook (run on A100)
├── training_results.png             # Training plots (reward + loss + before/after)
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Package config
├── requirements.txt                 # Runtime dependencies
├── Dockerfile                       # Container for HF Spaces deployment
├── inference.py                     # Standalone OpenEnv inference runner
├── demo.py                          # Local demo script
│
├── opensecops_env/                  # Core Python package
│   ├── __init__.py                  # Package init + version
│   ├── env.py                       # ⭐ Core environment (reset/step/state)
│   ├── grader.py                    # ⭐ Episode grader → [0, 1] score
│   ├── models.py                    # Data models (SecOpsAction, Observation, etc.)
│   ├── client.py                    # OpenEnv client wrapper
│   ├── inference.py                 # Inference utilities
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── task_definitions.py     # ⭐ 4 task configs (easy→hard)
│   └── server/
│       ├── __init__.py
│       └── app.py                  # ⭐ FastAPI server + dashboard + SSE streams
│
├── training/
│   ├── train_grpo.py               # Standalone GRPO training script
│   └── plot_rewards.py             # Generate training_results.png
│
├── tests/
│   └── test_opensecops.py          # 33 unit tests
│
├── docs/                           # Internal documentation
│   ├── DASHBOARD_GUIDE.md          # Plain-English dashboard explanation
│   ├── TECHNICAL_ANALYSIS.md       # Full pipeline + theme alignment
│   ├── analysis_and_next_steps.md  # Session notes
│   ├── code_explainer.md           # This file
│   └── walkthrough.md              # Development walkthrough
```

---

## Core Environment: `opensecops_env/env.py`

### Class: `OpenSecOpsEnv`

The main OpenEnv-compliant environment. Implements `reset()`, `step()`, and `state`.

```python
env = OpenSecOpsEnv()
obs = env.reset("hard_data_exfiltration")  # returns SecOpsObservation
obs, reward, done, info = env.step(SecOpsAction(
    action_type="query_logs",
    parameters={"service": "db"}
))
result = grade(env.state.to_dict())
```

**Key internal state:**
- `env._hidden: HiddenState` — ground truth (true_root_cause, affected_services, attack_progress, noise_level)
- `env._metrics: dict[str, ServiceMetrics]` — current CPU/mem/latency/error_rate per service
- `env._rng: random.Random` — seeded RNG; overridden per episode for variety
- `env._task_cfg: dict` — full task config from task_definitions.py
- `env._state: EpisodeState` — tracks investigation_actions, mitigation_actions, step_count, done

**Reward logic** (inside `env.step()`):
- Investigating the wrong service: `-0.05`
- Investigating an affected service (logs/scan): `+0.20` to `+0.30`
- Correct mitigation on affected service: `+0.50`
- Wrong mitigation/harmful action: `-0.10` to `-0.50`
- Correct final diagnosis: `+1.00`
- Wrong final diagnosis: `-1.00`
- Per-step cost: `-0.02`

---

## Task Definitions: `opensecops_env/tasks/task_definitions.py`

4 tasks with fixed seeds (overridden per episode by `_randomise_env_seed()`):

| ID | Difficulty | Seed | Noise | Affected Services | Correct Label |
|----|-----------|------|-------|-------------------|---------------|
| `easy_memory_leak` | Easy | 42 | 5% | auth | `infra_failure:memory_leak` |
| `medium_ddos_cascade` | Medium | 123 | 25% | gateway, api | `cyber_attack:ddos` |
| `medium_hard_bad_deployment` | Med-Hard | 456 | 35% | api, cache | `misconfiguration:bad_config` |
| `hard_data_exfiltration` | Hard | 789 | 55% | db, auth | `cyber_attack:data_exfiltration` |

Each task config includes: `initial_metrics`, `initial_alerts`, `initial_logs`, `topology`, `correct_mitigations`, `attack_progress_start`.

---

## Grader: `opensecops_env/grader.py`

```python
def grade(episode_state: dict) -> GradeResult:
    score = (
        0.5 * diagnosis_correct        # Was the final label correct?
      + 0.3 * action_efficiency        # Were actions targeted? Or scattered?
      + 0.2 * investigation_quality    # Did agent query/scan affected services?
    )
```

- `diagnosis_correct`: 1.0 if exact match, 0.5 if correct category, 0.0 if wrong
- `action_efficiency`: `0.7 * mitigation_recall + 0.3 * step_bonus`
- `investigation_quality`: fraction of affected services that were investigated

Score is clamped to `[0.01, 0.99]`.

---

## Multi-Agent System: `opensecops_env/server/app.py`

### Class: `MultiAgentSecOpsEnv`

Wraps `OpenSecOpsEnv` with two agents sharing the same environment state.

```python
ma_env = MultiAgentSecOpsEnv()
state = ma_env.reset("hard_data_exfiltration")

# Red (Attacker) acts first
state, red_reward, done, info = ma_env.red_step()  # heuristic auto

# Blue (Defender) acts
action = SecOpsAction(action_type="query_logs", parameters={"service": "db"})
state, blue_reward, done, info = ma_env.blue_step(action)
```

### Red Agent Strategy (`_heuristic_red_action`)

Adaptive 5-tier theory-of-mind strategy:
1. **Counter-investigate**: If Blue queried service X in last 3 steps → plant false alert on service Y
2. **Amplify**: If cyber_attack and attack_progress < 0.85 and Blue hasn't isolated → amplify
3. **Spread**: Spread to services Blue hasn't investigated yet via topology graph
4. **Corrupt**: Spike metrics on healthy services Blue has already looked at (plant doubt)
5. **Inject noise**: Default — add misleading log entries

### Class: `CurriculumManager`

```python
_curriculum.record_score(task_id, score)  # Called after every episode
_curriculum.current_level                 # 1-5
_curriculum.episode_count                 # total episodes this session
```

Level-up logic: rolling window of last 5 episodes for current level. If avg >= threshold → `current_level += 1`.

---

## SSE Streams: `/demo/stream` and `/battle/stream`

Both endpoints return `text/event-stream` with JSON events:

**Agent stream events:**
- `reset` — initial state + config
- `step` — action taken, reward, observation update, raw AI JSON
- `grade` — final scores + curriculum level
- `error` — exception message

**Battle stream events:**
- `battle_reset` — initial state
- `red_step` — attacker action + damage
- `blue_step` — defender action + reward + AI output
- `battle_end` — final scores + winner + curriculum level

---

## Live AI Integration: `_query_ai_model()`

```python
async def _query_ai_model(endpoint, obs_dict, step) -> Optional[SecOpsAction]:
    # Build text prompt from observation
    prompt = _obs_to_text(obs_dict, step)
    
    # POST to HF Inference Endpoint
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 128, "temperature": 0.3, "return_full_text": False}
    }
    headers = {"Authorization": f"Bearer {_HF_API_TOKEN}"}
    
    # Parse response (handles multiple output formats from the model)
    return _parse_ai_action(response_text)
```

**Auth:** Set `HF_TOKEN` in `.env` file. Auto-loaded via `python-dotenv` at startup.

**Debug:** GET `http://localhost:8000/debug/ai` to test live endpoint.

**Fallback:** If endpoint call fails, falls back to deterministic heuristic playbook (never crashes dashboard).

---

## Key Configuration

### `.env` (gitignored)
```
HF_TOKEN=hf_xxxx
TRAINED_MODEL_ENDPOINT=https://xxx.endpoints.huggingface.cloud  # optional override
```

### `requirements.txt`
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
httpx>=0.27.0
python-dotenv>=1.0.0
openenv-core>=0.2.0
```

### Running locally
```bash
cd incident-ai
.venv/bin/uvicorn opensecops_env.server.app:app --host 0.0.0.0 --port 8000 --reload
open http://localhost:8000/dashboard
```

---

## Training Pipeline

### GRPO Training (notebook: `colab_training (2).ipynb`)

```python
# Reward function — wraps the environment
def secops_reward_fn(prompts, completions, **kwargs):
    for completion, task_id in zip(completions, task_ids):
        action = parse_action(completion)
        env.reset(task_id)
        _, reward, _, _ = env.step(action)
        rewards.append(float(reward) - 0.02)  # step cost
    return rewards

# Trainer config
GRPOConfig(
    num_generations=4,       # 4 candidate responses per observation
    max_new_tokens=128,
    temperature=0.9,         # High temp for exploration during training
    learning_rate=2e-5,
)
```

**Model:** Qwen2.5-7B-Instruct + Unsloth 4-bit + LoRA (r=16)  
**Merge:** `model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit")`  
**Output:** `SapphireGaze429/opensecops-qwen2.5-7b-grpo`

---

## Tests: `tests/test_opensecops.py`

33 tests covering:
- Environment reset/step API contract
- All 4 task configs
- All 9 action types
- Reward bounds
- Grader formula correctness
- Partial diagnosis credit (category match)

```bash
pytest tests/ -v  # all 33 should pass
```
