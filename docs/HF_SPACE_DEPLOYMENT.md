# HuggingFace Space Deployment Guide
## Step-by-Step: Pushing OpenSecOpsEnv to HF Spaces

---

## What You're Deploying

A **Docker-based HF Space** that runs the FastAPI server + live dashboard.

- SDK: `docker`
- Port: `8000`
- The Space serves the dashboard at `/dashboard` and the environment API at the root
- Judges will pull the environment from this URL

---

## Prerequisites

- HuggingFace account: `SapphireGaze429`
- HF CLI installed locally
- `HF_TOKEN` (your write token) — already in `.env`
- Space already exists OR you'll create one now

---

## Step 1 — Log in to HF CLI

```bash
huggingface-cli login
```

Paste your token when prompted.

---

## Step 2 — Verify your Space exists

Go to: https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training

If the Space already exists (you can see a dashboard), skip to Step 3.

If it **doesn't exist yet**, create it:
```bash
huggingface-cli repo create opensecops-grpo-training --type space --space-sdk docker
```

---

## Step 3 — Check your README.md has the correct HF Space header

The top of `README.md` must have this YAML block (already in place):

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
  - multi-agent
  - grpo
---
```

✅ This is already correctly set.

---

## Step 4 — Add your HF_TOKEN as a Space Secret

**This is the most important step.** The live AI inference requires your token.

1. Go to: https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training/settings
2. Scroll to **"Repository secrets"**
3. Click **"New secret"**
4. Name: `HF_TOKEN`
5. Value: 
6. Click Save

> Without this, the Space will fall back to heuristic playbooks (still works for demo, but not live AI).

---

## Step 5 — Check .gitignore / .dockerignore

Make sure `.env` is gitignored (it is already).
Make sure `training_results.png` is NOT gitignored (it needs to be pushed).

Check:
```bash
cat .gitignore | grep training_results
cat .dockerignore | grep training_results
```

If either shows `training_results.png` as excluded, remove that line.

---

## Step 6 — Push to HF Space

HF Spaces work like a git repo. Push your code:

```bash
cd /Users/thapan/incedent-ai/incident-ai

# Add HF Space as remote (if not already done)
git remote add space https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training

# OR if remote already exists, update it:
git remote set-url space https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training

# Push main branch to the Space
git push space main
```

If it asks for username/password:
- Username: `SapphireGaze429`
- Password: 

---

## Step 7 — Wait for Build

HF Spaces will:
1. Detect the `Dockerfile`
2. Build the Docker image (takes 2-4 minutes)
3. Start the container
4. Expose port 8000

Watch the build logs at:
https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training → "Building" tab

---

## Step 8 — Verify the Space is Working

Once the build is green, open:
```
https://SapphireGaze429-opensecops-grpo-training.hf.space/dashboard
```

Or HF will show the dashboard embedded.

Also test the AI endpoint:
```
https://SapphireGaze429-opensecops-grpo-training.hf.space/debug/ai
```

Expected response:
```json
{
  "status_code": 200,
  "token_set": true,
  "endpoint": "https://hk5m5hadqtjiqg53...",
  "response": [{"generated_text": "..."}]
}
```

---

## Step 9 — Update links in README.md

Open `README.md` and make sure these links are correct (they should be already):

```markdown
[![HF Space](https://img.shields.io/badge/...)](https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training)
```

And in the All Resources table:
```markdown
| 🤗 HF Space (Live Demo) | https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training |
```

---

## Step 10 — Submit the URL to Hackathon

The URL judges need is:
```
https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training
```

This is your **environment URL** for the submission form.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| Build fails: `cannot find module openenv-core` | Check `requirements.txt` has `openenv-core>=0.2.0` |
| Dashboard loads but shows "AI timeout" | Add `HF_TOKEN` secret in Space settings (Step 4) |
| `git push space main` rejected | Run `huggingface-cli login` first |
| Space shows old code after push | Hard-refresh browser, check HF build logs |
| `training_results.png` not showing in README | Make sure it's in root dir and NOT in .gitignore |
| Port error in build logs | Confirm `app_port: 8000` in README.md YAML header |

---

## What Judges Will See

When judges visit your Space URL, they see:
1. **The README** embedded at the top (your full writeup + plots)
2. **The live dashboard** running in the iframe below the README
3. They can click "Run Episode" and watch your AI do real inference live

---

## Quick Reference — All Your URLs

| What | URL |
|------|-----|
| HF Space (submit this) | https://huggingface.co/spaces/SapphireGaze429/opensecops-grpo-training |
| Trained Model | https://huggingface.co/SapphireGaze429/opensecops-qwen2.5-7b-grpo |
| Live Dashboard | https://SapphireGaze429-opensecops-grpo-training.hf.space/dashboard |
| Debug AI | https://SapphireGaze429-opensecops-grpo-training.hf.space/debug/ai |
| Inference Endpoint | https://hk5m5hadqtjiqg53.us-east-1.aws.endpoints.huggingface.cloud |
