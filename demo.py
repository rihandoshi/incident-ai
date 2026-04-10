"""
demo.py - Walk through a complete OpenSecOpsEnv episode via HTTP calls.

HOW TO USE
----------
Step 1. Open a SEPARATE terminal and start the server:
            .venv/Scripts/uvicorn.exe opensecops_env.server.app:app --port 8000

Step 2. Keep that terminal open, then in THIS terminal run:
            python demo.py            (runs easy task)
            python demo.py medium     (DDoS task)
            python demo.py hard       (data exfiltration task)
"""

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Colour helpers (Windows PowerShell supports ANSI with VT sequences enabled)
# ---------------------------------------------------------------------------
def bold(s):   return "\033[1m"  + str(s) + "\033[0m"
def green(s):  return "\033[92m" + str(s) + "\033[0m"
def red(s):    return "\033[91m" + str(s) + "\033[0m"
def yellow(s): return "\033[93m" + str(s) + "\033[0m"
def cyan(s):   return "\033[96m" + str(s) + "\033[0m"
def dim(s):    return "\033[2m"  + str(s) + "\033[0m"

def sep(title=""):
    if title:
        pad = max(2, (58 - len(title)) // 2)
        print("\n" + bold("=" * pad + "  " + title + "  " + "=" * pad))
    else:
        print(bold("=" * 60))

# ---------------------------------------------------------------------------
# HTTP helpers (no third-party deps needed)
# ---------------------------------------------------------------------------
def post(path, payload=None):
    url  = BASE_URL + path
    data = json.dumps(payload or {}).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

def get(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as resp:
        return json.loads(resp.read())

# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------
def print_observation(obs):
    print("\n  " + bold("ALERTS") + " (" + str(len(obs["alerts"])) + " total):")
    for a in obs["alerts"]:
        sev    = a.get("severity", "?").upper()
        colour = red if sev == "CRITICAL" else yellow if sev == "WARNING" else dim
        svc    = a.get("service", "?")
        typ    = a.get("type", "?")
        msg    = a.get("message", "")
        print("    " + colour("[" + sev + "]") +
              "  " + f"{svc:<10}" + "  " + f"{typ:<22}" + "  " + dim(msg))

    print("\n  " + bold("METRICS") + ":")
    for svc, m in obs["metrics"].items():
        cpu = m["cpu"];   mem = m["memory"]
        lat = m["latency"]; err = m["error_rate"]
        cpu_c = red(f"{cpu:5.1f}%")  if cpu > 80 else yellow(f"{cpu:5.1f}%")  if cpu > 60 else f"{cpu:5.1f}%"
        mem_c = red(f"{mem:5.1f}%")  if mem > 85 else yellow(f"{mem:5.1f}%")  if mem > 70 else f"{mem:5.1f}%"
        lat_c = red(f"{lat:6.0f}ms") if lat > 500 else yellow(f"{lat:6.0f}ms") if lat > 200 else f"{lat:6.0f}ms"
        err_c = red(f"{err:5.2f}%")  if err > 10  else yellow(f"{err:5.2f}%")  if err > 5   else f"{err:5.2f}%"
        print("    " + f"{svc:<12}" + "cpu=" + cpu_c +
              "  mem=" + mem_c + "  lat=" + lat_c + "  err=" + err_c)

    print("\n  " + bold("LOGS") + " (recent window -- may contain misleading noise):")
    for line in obs.get("logs", []):
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        colour = red if ("ERROR" in line or "CRIT" in line) else yellow if "WARN" in line else dim
        print("    " + colour(safe_line))

    print("\n  " + bold("TOPOLOGY") + " (service dependencies):")
    for svc, deps in obs.get("topology", {}).items():
        print("    " + f"{svc:<12}" + "--> " + str(deps))


def print_step_result(step_num, action_type, params, reward, done, result_msg):
    r_str    = green("+" + f"{reward:.3f}") if reward >= 0 else red(f"{reward:.3f}")
    done_str = red("[DONE]") if done else dim("[continuing...]")
    print("\n  " + bold("Step " + str(step_num)) +
          "  " + cyan(action_type) + "(" + json.dumps(params) + ")")
    print("  Reward : " + r_str + "   " + done_str)
    safe_msg = result_msg.encode("ascii", errors="replace").decode("ascii")
    print("  Result : " + safe_msg)


def print_grade(g):
    score  = g["score"]
    colour = green if score >= 0.7 else yellow if score >= 0.4 else red
    print("\n  " + bold("Final Score      :") + " " + colour(bold(f"{score:.4f}")) + " / 1.0")
    print("  Diagnosis        : " + str(g["diagnosis_correct"]) +
          "   (1.0 = exact match, 0.5 = right category, 0.0 = wrong)")
    print("  Action Efficiency: " + f"{g['action_efficiency']:.3f}" +
          "   (correct mitigations taken / total, adjusted for steps used)")
    print("  Investigation    : " + f"{g['investigation_quality']:.3f}" +
          "   (fraction of affected services you actually investigated)")
    print("\n  Correct label    : " + green(g["details"]["correct_label"]))
    print("  Submitted label  : " + g["details"]["submitted_label"])


# ---------------------------------------------------------------------------
# Optimal playbooks (what a perfect agent would do on each task)
# ---------------------------------------------------------------------------
PLAYBOOKS = {
    "easy": {
        "task_id": "easy_memory_leak",
        "steps": [
            ("inspect_metrics", {},                               "Look at ALL service metrics at once - survey the damage"),
            ("query_logs",      {"service": "auth"},              "auth has high memory - check its logs"),
            ("inspect_metrics", {"service": "auth"},              "Drill into auth metrics specifically"),
            ("restart_service", {"service": "auth"},              "Memory leak -> restarting clears the heap"),
            ("submit_diagnosis",{"label": "infra_failure:memory_leak"}, "Submit final root cause label"),
        ],
    },
    "medium": {
        "task_id": "medium_ddos_cascade",
        "steps": [
            ("inspect_metrics",   {},                             "Survey all services - many are degraded"),
            ("query_logs",        {"service": "gateway"},         "Gateway logs reveal attacking IP addresses"),
            ("run_security_scan", {"target": "api"},              "Security scan on api confirms DDoS pattern"),
            ("block_ip",          {"ip": "203.0.113.45"},         "Block attacker IP 1 - traffic drops"),
            ("block_ip",          {"ip": "198.51.100.12"},        "Block attacker IP 2 - more traffic drops"),
            ("scale_service",     {"service": "api","replicas":5},"Scale api out to absorb remaining load"),
            ("submit_diagnosis",  {"label": "cyber_attack:ddos"}, "Submit final root cause label"),
        ],
    },
    "hard": {
        "task_id": "hard_data_exfiltration",
        "steps": [
            ("inspect_metrics",   {},                                   "Survey - IGNORE the cache alert, it is a decoy"),
            ("query_logs",        {"service": "db"},                    "DB logs: 4GB outbound transfer to 10.0.0.99"),
            ("query_logs",        {"service": "auth"},                  "Auth logs: privilege escalation by reports_bot"),
            ("run_security_scan", {"target": "db"},                     "Scan db: confirms active exfiltration"),
            ("run_security_scan", {"target": "auth"},                   "Scan auth: confirms compromised account"),
            ("isolate_service",   {"service": "db"},                    "Isolate db - cuts off exfiltration channel"),
            ("block_ip",          {"ip": "10.0.0.99"},                  "Block the external destination IP"),
            ("submit_diagnosis",  {"label": "cyber_attack:data_exfiltration"}, "Submit final root cause label"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "easy"
    if difficulty not in PLAYBOOKS:
        print("Unknown difficulty '" + difficulty + "'. Choose: easy / medium / hard")
        sys.exit(1)

    playbook = PLAYBOOKS[difficulty]
    task_id  = playbook["task_id"]

    # ---------- health check ----------
    sep("SERVER HEALTH CHECK")
    try:
        health = get("/health")
        print(green("  Server is online: " + str(health)))
    except (urllib.error.URLError, OSError):
        print(red("  ERROR: Cannot reach server at http://localhost:8000"))
        print(yellow("  Fix: open a separate terminal and run:"))
        print(yellow("       .venv/Scripts/uvicorn.exe opensecops_env.server.app:app --port 8000"))
        sys.exit(1)

    # ---------- reset ----------
    sep("RESET  --  task: " + task_id)
    reset_resp = post("/reset", {"task_id": task_id})
    obs = reset_resp["observation"]
    print("  New episode started at time_step=" + str(obs["time_step"]))
    print_observation(obs)

    # ---------- steps ----------
    cumulative_reward = 0.0
    for i, (action_type, params, explanation) in enumerate(playbook["steps"], 1):
        sep("STEP " + str(i) + "  --  " + explanation)
        resp   = post("/step", {"action_type": action_type, "parameters": params})
        obs    = resp["observation"]
        reward = resp["reward"]
        done   = resp["done"]
        result = obs["last_action_result"]
        cumulative_reward += reward
        print_step_result(i, action_type, params, reward, done, result)
        if not done:
            print_observation(obs)
        else:
            break

    # ---------- grade ----------
    sep("GRADER  --  Final Score Breakdown")
    grade_resp = post("/grade")
    print_grade(grade_resp)

    # ---------- reveal hidden state ----------
    sep("INTERNAL STATE  (what the env hid from the agent)")
    state  = get("/state")["state"]
    hidden = state["hidden_state"]
    print("  True root cause : " + bold(red(hidden["true_root_cause"])))
    print("  Subtype         : " + bold(red(hidden["subtype"])))
    print("  Affected svcs   : " + str(hidden["affected_services"]))
    print("  Attack progress : " + str(hidden["attack_progress"]) +
          "  (0=start, 1=full)")
    print("  Noise level     : " + str(hidden["noise_level"]) +
          "  (fraction of misleading signals injected)")
    print("  Steps taken     : " + str(state["step_count"]))
    print("  Cumul. reward   : " + str(round(cumulative_reward, 3)))

    sep()
    print(green(bold("  Done!  Re-run with:  python demo.py easy | medium | hard")))
    print()

if __name__ == "__main__":
    main()
