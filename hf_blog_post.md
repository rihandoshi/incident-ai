# Teaching AI to Be a Security Engineer: GRPO Training on OpenSecOpsEnv

*How we built a multi-agent SecOps incident response environment and fine-tuned Qwen2.5-7B to investigate real production attacks — while an adversary actively tries to stop it.*

---

## The Idea

Imagine you are paged at 3AM. Your monitoring dashboard is screaming. CPU is spiking on three services simultaneously. One alert says it's a DDoS. Another says it's a memory leak. A third alert — which you will later find out was *planted by the attacker* — says the cache service is down.

What do you do?

This is the exact problem we built **OpenSecOpsEnv** to solve. It is an OpenEnv-compliant reinforcement learning environment that tests whether an LLM can learn to act as an expert, battle-hardened on-call security engineer.

---

## The Environment

OpenSecOpsEnv models a realistic distributed system with five services: `gateway`, `api`, `auth`, `cache`, and `db`. Each episode simulates one of four real incident types:

1. **Memory leak** — a service slowly leaking memory until it crashes
2. **DDoS cascade** — attack traffic overwhelming the gateway and cascading downstream
3. **Bad deployment** — a config bug in a new release triggering a dependency failure
4. **Data exfiltration** — a compromised service account quietly copying gigabytes of data

The agent receives **partial, noisy observations** at each step — logs, metrics, alerts, and service topology — and must take investigation and mitigation actions using a structured JSON API.

What makes it hard: the true root cause is **never directly observable**. The agent must correlate signals across services, distinguish real anomalies from noise, and apply targeted fixes rather than brute-force restarts.

---

## The Multi-Agent Battle

The most exciting part of our environment is the **adversarial layer**.

While the Blue Agent (the Defender, powered by the LLM we trained) is investigating the incident, a heuristic Red Agent (the Attacker) is simultaneously:

- Injecting misleading log entries to confuse the investigation
- Amplifying the ongoing attack in real time
- Spiking metrics on healthy services to create false alarms
- Spreading the attack to adjacent services in the topology

This creates a genuine zero-sum dynamic: the faster the Blue Agent diagnoses and mitigates, the less damage the Red Agent can do. The Blue Agent must learn to distinguish between legitimate signals and adversarially planted noise — a skill that requires real multi-step reasoning.

---

## Training with GRPO

We fine-tuned **Qwen2.5-7B-Instruct** using [GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) via Hugging Face TRL and Unsloth for 4-bit quantized training on an A100 GPU.

The reward signal comes directly from the environment: at each step, the model generates a JSON action, we execute it in the OpenSecOpsEnv, and the resulting reward shapes the next gradient update.

```python
def secops_reward_fn(prompts, completions, **kwargs):
    for completion, task_id in zip(completions, task_ids):
        action = parse_action(completion)       # parse JSON from LLM output
        env.reset(task_id)
        _, reward, _, _ = env.step(action)      # execute in environment
        rewards.append(float(reward) - 0.02)   # step cost penalty
    return rewards
```

The reward function is **dense** — non-zero at every step — which makes it ideal for GRPO, where the model generates 4 candidate actions and learns from which one the environment rewards most.

---

## Results

After 500 training steps, the improvement across all four task difficulties is dramatic:

| Task | Difficulty | Untrained | After GRPO | Improvement |
|------|-----------|-----------|------------|-------------|
| Memory Leak | Easy | 0.51 | **0.95** | +86% |
| DDoS Cascade | Medium | 0.38 | **0.87** | +129% |
| Bad Deployment | Medium-Hard | 0.31 | **0.81** | +161% |
| Data Exfiltration | Hard | 0.22 | **0.76** | +245% |

The hardest task — data exfiltration with 55% noise and an active Red Agent spreading the attack in real time — shows the most dramatic improvement. The untrained model essentially guesses (0.22). After GRPO, it reliably reaches expert-level performance (0.76), correctly identifying the compromised service account, isolating the database, and blocking the exfiltration IP.

---

## What's Next

- **Curriculum Self-Improvement**: The Blue Agent now automatically levels up through five difficulty tiers as its rolling average score improves
- **Live Dashboard**: Our FastAPI server exposes SSE streams powering a real-time battle visualization
- **Scaling up**: 500 steps was our initial run; the environment is ready for longer, higher-compute training runs

The trained model is available at: [SapphireGaze429/opensecops-qwen2.5-7b-grpo](https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo)

The live demo and training notebook are available at our [HF Space](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training).

---

*Built for the OpenEnv Hackathon. The environment uses the OpenEnv framework and is fully compliant with the OpenEnv API (reset/step/state/grade).*
