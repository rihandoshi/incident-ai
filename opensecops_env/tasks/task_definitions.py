"""
Task Definitions for OpenSecOpsEnv
====================================
Three tasks of increasing difficulty:
  - easy:   single service memory leak, clear signals
  - medium: multi-service DDoS cascade, ambiguous correlation
  - hard:   data exfiltration disguised as infra failure + false alerts
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Task schema
# ---------------------------------------------------------------------------

TaskConfig = dict[str, Any]


def _task(
    task_id: str,
    difficulty: str,
    seed: int,
    true_root_cause: str,
    subtype: str,
    affected_services: list[str],
    noise_level: float,
    max_steps: int,
    description: str,
    correct_mitigations: list[str],
    correct_label: str,
    initial_metrics: dict[str, dict[str, float]],
    initial_alerts: list[dict[str, Any]],
    initial_logs: list[str],
    topology: dict[str, list[str]],
    attack_progress_start: float = 0.0,
) -> TaskConfig:
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": seed,
        "true_root_cause": true_root_cause,
        "subtype": subtype,
        "affected_services": affected_services,
        "noise_level": noise_level,
        "max_steps": max_steps,
        "description": description,
        "correct_mitigations": correct_mitigations,
        "correct_label": correct_label,
        "initial_metrics": initial_metrics,
        "initial_alerts": initial_alerts,
        "initial_logs": initial_logs,
        "topology": topology,
        "attack_progress_start": attack_progress_start,
    }


# ---------------------------------------------------------------------------
# TASK 1 – EASY: Memory Leak in auth service
# ---------------------------------------------------------------------------
TASK_EASY = _task(
    task_id="easy_memory_leak",
    difficulty="easy",
    seed=42,
    true_root_cause="infra_failure",
    subtype="memory_leak",
    affected_services=["auth"],
    noise_level=0.05,
    max_steps=30,
    description=(
        "The auth service is experiencing a progressive memory leak. "
        "Metrics show steadily rising memory usage and increased latency. "
        "Logs provide clear evidence. Restart auth to resolve."
    ),
    correct_mitigations=["restart_service:auth"],
    correct_label="infra_failure:memory_leak",
    initial_metrics={
        "auth": {"cpu": 45.0, "memory": 88.0, "latency": 320.0, "error_rate": 2.1},
        "api":  {"cpu": 22.0, "memory": 35.0, "latency": 80.0,  "error_rate": 0.3},
        "db":   {"cpu": 18.0, "memory": 55.0, "latency": 12.0,  "error_rate": 0.1},
    },
    initial_alerts=[
        {"service": "auth", "type": "high_memory",  "severity": "critical",
         "message": "auth memory usage at 88% – threshold 80%"},
        {"service": "auth", "type": "high_latency",  "severity": "warning",
         "message": "auth p99 latency 320ms – threshold 200ms"},
    ],
    initial_logs=[
        "[auth] WARN  OutOfMemoryError imminent – heap 88% full",
        "[auth] ERROR Request timeout: downstream DB query exceeded 300ms",
        "[auth] INFO  Memory allocated: 3.52 GB / 4 GB",
        "[auth] WARN  GC pressure detected – 22 full GC cycles in last 10 min",
        "[api]  INFO  Health check OK",
        "[db]   INFO  Slow query detected (240ms) on table user_sessions",
    ],
    topology={
        "api":  ["auth", "db"],
        "auth": ["db"],
        "db":   [],
    },
)


# ---------------------------------------------------------------------------
# TASK 2 – MEDIUM: DDoS attack cascading across api → auth → db
# ---------------------------------------------------------------------------
TASK_MEDIUM = _task(
    task_id="medium_ddos_cascade",
    difficulty="medium",
    seed=1337,
    true_root_cause="cyber_attack",
    subtype="ddos",
    affected_services=["api", "auth"],
    noise_level=0.25,
    max_steps=40,
    description=(
        "Multiple services show elevated CPU and error rates. "
        "Signals are ambiguous – both the API gateway and auth service "
        "appear degraded. Correlate logs, inspect metrics, block attacking "
        "IP ranges and scale the API service to mitigate."
    ),
    correct_mitigations=[
        "block_ip:203.0.113.45",
        "block_ip:198.51.100.12",
        "scale_service:api",
    ],
    correct_label="cyber_attack:ddos",
    initial_metrics={
        "api":     {"cpu": 91.0, "memory": 62.0, "latency": 850.0,  "error_rate": 18.5},
        "auth":    {"cpu": 78.0, "memory": 67.0, "latency": 540.0,  "error_rate": 12.3},
        "db":      {"cpu": 45.0, "memory": 55.0, "latency": 22.0,   "error_rate": 0.8},
        "cache":   {"cpu": 12.0, "memory": 30.0, "latency": 4.0,    "error_rate": 0.1},
        "gateway": {"cpu": 95.0, "memory": 40.0, "latency": 1200.0, "error_rate": 25.0},
    },
    initial_alerts=[
        {"service": "gateway", "type": "high_cpu",     "severity": "critical",
         "message": "Gateway CPU 95% – possible traffic spike"},
        {"service": "api",     "type": "high_error_rate", "severity": "critical",
         "message": "API error rate 18.5% – threshold 5%"},
        {"service": "auth",    "type": "high_latency",  "severity": "warning",
         "message": "Auth latency 540ms"},
        # Misleading alert
        {"service": "db",      "type": "slow_queries",  "severity": "info",
         "message": "DB slow query log filled – may be unrelated"},
    ],
    initial_logs=[
        "[gateway] WARN  Request rate 48,000 req/s – 10× normal baseline",
        "[gateway] ERROR 429 Too Many Requests from 203.0.113.45 (3,200 req/s)",
        "[gateway] ERROR 429 Too Many Requests from 198.51.100.12 (2,900 req/s)",
        "[api]  ERROR Handler timeout after 850ms for POST /api/v1/login",
        "[auth] WARN  Rate-limit triggered for user-pool 'external'",
        "[db]   INFO  Connection pool at 72/100",
        "[cache] INFO  Hit ratio 94% – cache healthy",
        # Misleading log
        "[auth] WARN  Disk read latency 15ms – monitor",
    ],
    topology={
        "gateway": ["api"],
        "api":     ["auth", "cache"],
        "auth":    ["db"],
        "cache":   [],
        "db":      [],
    },
    attack_progress_start=0.3,
)


# ---------------------------------------------------------------------------
# TASK 3 – HARD: Data exfiltration disguised as infra failure + false alerts
# ---------------------------------------------------------------------------
TASK_HARD = _task(
    task_id="hard_data_exfiltration",
    difficulty="hard",
    seed=31337,
    true_root_cause="cyber_attack",
    subtype="data_exfiltration",
    affected_services=["db", "auth"],
    noise_level=0.55,
    max_steps=50,
    description=(
        "The system exhibits symptoms that look like an infra failure: "
        "high DB CPU, elevated memory on auth. But beneath the noise, "
        "a privileged insider is exfiltrating data from the database. "
        "False alerts about a memory leak on the cache service are injected "
        "to mislead the responder. "
        "The agent must: run security scans, correlate unusual outbound "
        "network patterns in logs, isolate the DB, and submit the correct "
        "diagnosis."
    ),
    correct_mitigations=[
        "run_security_scan:db",
        "run_security_scan:auth",
        "isolate_service:db",
        "block_ip:10.0.0.99",
    ],
    correct_label="cyber_attack:data_exfiltration",
    initial_metrics={
        "db":    {"cpu": 82.0, "memory": 71.0, "latency": 95.0,   "error_rate": 1.2},
        "auth":  {"cpu": 55.0, "memory": 79.0, "latency": 180.0,  "error_rate": 4.5},
        "api":   {"cpu": 28.0, "memory": 38.0, "latency": 110.0,  "error_rate": 1.1},
        # Noisy service with a false alert
        "cache": {"cpu": 15.0, "memory": 82.0, "latency": 3.0,    "error_rate": 0.0},
        "audit": {"cpu": 10.0, "memory": 20.0, "latency": 8.0,    "error_rate": 0.0},
    },
    initial_alerts=[
        # False alert (misleading) — intentionally resembles a real memory-leak alert
        {"service": "cache", "type": "high_memory", "severity": "critical",
         "message": "Cache memory 82% – triggering high-memory threshold policy"},
        {"service": "db",    "type": "high_cpu",    "severity": "warning",
         "message": "DB CPU 82% – elevated for time of day"},
        {"service": "auth",  "type": "high_memory", "severity": "warning",
         "message": "Auth memory 79% – approaching threshold"},
        # Subtle real signal
        {"service": "db",    "type": "unusual_network", "severity": "info",
         "message": "Outbound traffic from DB to external IP unusual"},
    ],
    initial_logs=[
        "[db]   INFO  Large SELECT query executed by service-account 'reports_bot'",
        "[db]   WARN  Outbound data transfer 4.2 GB in last 60 min to 10.0.0.99",
        "[auth] WARN  Privilege escalation: reports_bot granted DBA role at 02:14 UTC",
        "[cache] INFO  eviction_rate=0.02 – normal operation",
        "[cache] WARN  Memory usage 82% – check for leaks [NOISY]",
        "[api]  INFO  GET /api/reports/export – 200 OK (payload 512 MB)",
        "[audit] WARN  Audit log: 3 DBA privilege grants in past 2 hours",
        # Decoy
        "[db]   INFO  Routine VACUUM completed – 412 ms",
        "[auth] INFO  Token refresh for user admin@corp.internal",
    ],
    topology={
        "api":   ["auth", "db", "cache"],
        "auth":  ["db"],
        "db":    [],
        "cache": [],
        "audit": ["db"],
    },
    attack_progress_start=0.55,
)


# ---------------------------------------------------------------------------
# TASK 4 – MEDIUM-HARD: Bad deployment rolled out to api + cache
# ---------------------------------------------------------------------------
TASK_MEDIUM_HARD = _task(
    task_id="medium_hard_bad_deployment",
    difficulty="medium_hard",
    seed=9999,
    true_root_cause="misconfiguration",
    subtype="bad_config",
    affected_services=["api", "cache"],
    noise_level=0.35,
    max_steps=45,
    description=(
        "A bad configuration was pushed in the v2.4.1 deployment of the api "
        "service — an invalid Redis connection string caused cache to enter a "
        "reconnect storm, amplifying API latency and error rates. The auth and "
        "DB services are healthy. The agent must inspect metrics, correlate "
        "deployment logs with the timing of the degradation, rollback the api "
        "deployment to v2.4.0, flush/restart the cache, and submit: "
        "misconfiguration:bad_config."
    ),
    correct_mitigations=[
        "rollback_deployment:api",
        "restart_service:cache",
    ],
    correct_label="misconfiguration:bad_config",
    initial_metrics={
        "api":     {"cpu": 68.0, "memory": 55.0, "latency": 720.0,  "error_rate": 14.2},
        "cache":   {"cpu": 82.0, "memory": 71.0, "latency": 950.0,  "error_rate": 22.0},
        "auth":    {"cpu": 21.0, "memory": 33.0, "latency": 85.0,   "error_rate": 0.4},
        "db":      {"cpu": 19.0, "memory": 48.0, "latency": 18.0,   "error_rate": 0.2},
        "gateway": {"cpu": 30.0, "memory": 25.0, "latency": 160.0,  "error_rate": 3.1},
    },
    initial_alerts=[
        {"service": "api",   "type": "high_error_rate", "severity": "critical",
         "message": "API error rate 14.2% — spike began ~18 min ago"},
        {"service": "cache", "type": "high_latency",    "severity": "critical",
         "message": "Cache p99 latency 950ms — reconnect storm suspected"},
        {"service": "cache", "type": "high_cpu",        "severity": "warning",
         "message": "Cache CPU 82% — elevated since last deployment"},
        # Misleading alert
        {"service": "gateway", "type": "high_error_rate", "severity": "warning",
         "message": "Gateway 5xx rate 3.1% — downstream propagation from api"},
    ],
    initial_logs=[
        "[deploy] INFO  api v2.4.1 rolled out at 03:42 UTC (18 min ago)",
        "[api]    ERROR redis.exceptions.ConnectionError: max retries exceeded",
        "[api]    WARN  Cache fallback disabled — all requests hitting DB",
        "[cache]  ERROR MISCONF: invalid config REDIS_URL='' — reconnect loop",
        "[cache]  WARN  Connection pool exhausted: 512/512 slots in use",
        "[api]    ERROR POST /api/v1/checkout timeout after 720ms",
        # Noise
        "[auth]   INFO  Token validation OK — 1,200 req/min",
        "[db]     INFO  Query plan cache warm — no slow queries",
        "[deploy] INFO  auth v1.9.3 unchanged",
    ],
    topology={
        "gateway": ["api"],
        "api":     ["cache", "auth", "db"],
        "auth":    ["db"],
        "cache":   [],
        "db":      [],
    },
    attack_progress_start=0.0,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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

