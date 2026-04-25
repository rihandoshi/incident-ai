# 🧠 OpenEnv Hackathon — Judging & Expectations Guide

## 🚨 TL;DR (What You Actually Need to Do)
Build an environment where an LLM can **train and measurably improve at something meaningful**, then:
- Show **actual training**
- Provide **evidence (metrics, reward curves, comparisons)**
- Tell a **clear, compelling story**

A messy but ambitious project with real training evidence beats a polished but shallow one.

---

# ⚖️ Judging Criteria (Core Evaluation)

## 1. 🌟 Environment Innovation — 40%
- Is your environment **novel, creative, or challenging**?
- Does it **meaningfully test agent behavior**?
- Avoid overused ideas (grid worlds, chess clones, etc.)

## 2. 🎤 Storytelling & Presentation — 30%
- Clearly explain:
  - The problem
  - The environment
  - What the agent learned
- Demo should be engaging and easy to follow

## 3. 📈 Showing Improvement in Rewards — 20%
- Must prove learning happened
- Evidence:
  - Reward curves
  - Before vs after behavior
  - Baseline comparisons

## 4. ⚙️ Reward & Training Pipeline — 10%
- Reward logic should be coherent and hard to exploit
- Training should improve agent behavior

---

# 📦 Minimum Submission Requirements

- Use **OpenEnv (latest release)**
- Provide a **working training script** (Unsloth or HuggingFace TRL)
- Show **training evidence** (loss + reward plots)
- Submit:
  - Mini-blog OR
  - <2 min video OR
  - Slides
- Host on **Hugging Face Spaces**
- Provide a **README with problem, environment, results, links**

### Rules:
- One submission per team
- Submit environment URL
- No changes after deadline

---

# 🧪 What Judges Look For

## 🔬 Real Training
- Training must run against your environment
- Show learning with plots, metrics, comparisons

## 🧠 Reward Design
- Dense and informative rewards
- Hard to game
- Avoid simple binary rewards

## 🚀 Ambitious Problems
- Solve something LLMs struggle with
- Prefer underexplored domains

## 📊 Clear Results
- Label axes properly
- Save plots as images
- Show comparisons clearly

## 📖 Tell a Story
Your README should answer:
1. Problem
2. Environment
3. Results
4. Why it matters

## 🧹 Clean Engineering
- Use OpenEnv properly
- Follow Gym API (reset, step, state)
- Maintain clean architecture

---

# 🧭 Problem Selection Guidelines
- Reuse Round 1 idea only if it fits themes
- Build environment + reward model early
- Ensure alignment with judging criteria

---

# 🏁 Final Advice
- Be ambitious
- Show real learning
- Communicate clearly

Judges want projects that push the frontier of LLM training.
