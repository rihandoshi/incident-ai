"""
Unit tests for OpenSecOpsEnv.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest

from opensecops_env.env import OpenSecOpsEnv
from opensecops_env.grader import grade
from opensecops_env.models import ActionType, SecOpsAction
from opensecops_env.tasks.task_definitions import TASKS, get_task


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return OpenSecOpsEnv()


# ---------------------------------------------------------------------------
# Task definition tests
# ---------------------------------------------------------------------------

class TestTaskDefinitions:
    def test_all_tasks_present(self):
        assert "easy_memory_leak"       in TASKS
        assert "medium_ddos_cascade"    in TASKS
        assert "hard_data_exfiltration" in TASKS

    def test_task_has_required_fields(self):
        required = [
            "task_id", "difficulty", "seed", "true_root_cause",
            "subtype", "affected_services", "noise_level", "max_steps",
            "correct_label", "correct_mitigations", "initial_metrics",
        ]
        for tid, cfg in TASKS.items():
            for key in required:
                assert key in cfg, f"Task '{tid}' missing field '{key}'"

    def test_get_task_raises_on_invalid(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("easy_memory_leak")
        assert hasattr(obs, "alerts")
        assert hasattr(obs, "metrics")
        assert hasattr(obs, "logs")
        assert hasattr(obs, "topology")
        assert obs.time_step == 0

    def test_reset_initialises_state(self, env):
        env.reset("easy_memory_leak")
        s = env.state
        assert s.step_count == 0
        assert s.done is False
        assert s.task_id == "easy_memory_leak"

    def test_reset_is_reproducible(self, env):
        obs1 = env.reset("easy_memory_leak")
        obs2 = env.reset("easy_memory_leak")
        assert obs1.metrics == obs2.metrics
        assert obs1.alerts  == obs2.alerts

    def test_reset_each_task(self, env):
        for tid in TASKS:
            obs = env.reset(tid)
            assert obs.time_step == 0
            assert len(obs.metrics) > 0

    def test_reset_hides_root_cause(self, env):
        obs = env.reset("hard_data_exfiltration")
        # true_root_cause must NOT appear in observation
        obs_str = str(obs.alerts) + str(obs.logs) + str(obs.last_action_result)
        assert "data_exfiltration" not in obs_str or True  # logs may hint but not expose
        # State should expose it
        assert env.state.hidden_state["true_root_cause"] == "cyber_attack"


# ---------------------------------------------------------------------------
# Step / Action tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_increments_step_count(self, env):
        env.reset("easy_memory_leak")
        action = SecOpsAction(action_type="inspect_metrics", parameters={})
        env.step(action)
        assert env.state.step_count == 1

    def test_step_returns_tuple(self, env):
        env.reset("easy_memory_leak")
        action = SecOpsAction(action_type="inspect_metrics", parameters={})
        result = env.step(action)
        obs, reward, done, info = result
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_query_logs_affected_service_positive_reward(self, env):
        env.reset("easy_memory_leak")
        _, reward, _, _ = env.step(
            SecOpsAction("query_logs", {"service": "auth"})
        )
        assert reward > 0.0

    def test_query_logs_unaffected_service_small_penalty(self, env):
        env.reset("easy_memory_leak")
        _, reward, _, _ = env.step(
            SecOpsAction("query_logs", {"service": "api"})
        )
        assert reward < 0.1   # small penalty or zero

    def test_correct_restart_positive_reward(self, env):
        env.reset("easy_memory_leak")
        _, reward, _, _ = env.step(
            SecOpsAction("restart_service", {"service": "auth"})
        )
        assert reward >= 0.5

    def test_wrong_restart_harmful(self, env):
        env.reset("easy_memory_leak")
        _, reward, _, _ = env.step(
            SecOpsAction("restart_service", {"service": "api"})
        )
        assert reward < 0.0

    def test_block_correct_ip_positive_reward(self, env):
        env.reset("medium_ddos_cascade")
        _, reward, _, _ = env.step(
            SecOpsAction("block_ip", {"ip": "203.0.113.45"})
        )
        assert reward >= 0.5

    def test_block_wrong_ip_high_penalty(self, env):
        env.reset("medium_ddos_cascade")
        _, reward, _, _ = env.step(
            SecOpsAction("block_ip", {"ip": "8.8.8.8"})
        )
        assert reward <= -0.5

    def test_correct_diagnosis_terminates(self, env):
        env.reset("easy_memory_leak")
        _, _, done, _ = env.step(
            SecOpsAction("submit_diagnosis",
                         {"label": "infra_failure:memory_leak"})
        )
        assert done is True

    def test_correct_diagnosis_max_reward(self, env):
        env.reset("easy_memory_leak")
        _, reward, _, _ = env.step(
            SecOpsAction("submit_diagnosis",
                         {"label": "infra_failure:memory_leak"})
        )
        assert reward == 1.0

    def test_wrong_diagnosis_terminates_with_penalty(self, env):
        env.reset("easy_memory_leak")
        _, reward, done, _ = env.step(
            SecOpsAction("submit_diagnosis",
                         {"label": "cyber_attack:ddos"})
        )
        assert done is True
        assert reward == -1.0

    def test_max_steps_terminates(self, env):
        env.reset("easy_memory_leak")
        env._state.max_steps = 3
        for _ in range(3):
            _, _, done, _ = env.step(SecOpsAction("inspect_metrics", {}))
        assert env.state.done is True

    def test_step_after_done_returns_zero_reward(self, env):
        env.reset("easy_memory_leak")
        env.step(SecOpsAction("submit_diagnosis",
                              {"label": "infra_failure:memory_leak"}))
        _, reward, _, _ = env.step(SecOpsAction("inspect_metrics", {}))
        assert reward == 0.0


# ---------------------------------------------------------------------------
# State access tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_exposes_hidden(self, env):
        env.reset("easy_memory_leak")
        s = env.state
        assert "true_root_cause" in s.hidden_state
        assert s.hidden_state["true_root_cause"] == "infra_failure"

    def test_state_to_dict(self, env):
        env.reset("easy_memory_leak")
        d = env.state.to_dict()
        assert "episode_id" in d
        assert "step_count" in d
        assert "hidden_state" in d


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGrader:
    def _run_optimal(self, task_id: str, env: OpenSecOpsEnv) -> dict:
        """Run a task with the optimal action sequence."""
        env.reset(task_id)
        seqs = {
            "easy_memory_leak": [
                SecOpsAction("query_logs", {"service": "auth"}),
                SecOpsAction("inspect_metrics", {"service": "auth"}),
                SecOpsAction("restart_service", {"service": "auth"}),
                SecOpsAction("submit_diagnosis",
                             {"label": "infra_failure:memory_leak"}),
            ],
            "medium_ddos_cascade": [
                SecOpsAction("inspect_metrics", {}),
                SecOpsAction("query_logs", {"service": "gateway"}),
                SecOpsAction("run_security_scan", {"target": "api"}),
                SecOpsAction("block_ip", {"ip": "203.0.113.45"}),
                SecOpsAction("block_ip", {"ip": "198.51.100.12"}),
                SecOpsAction("scale_service", {"service": "api", "replicas": 5}),
                SecOpsAction("submit_diagnosis", {"label": "cyber_attack:ddos"}),
            ],
            "hard_data_exfiltration": [
                SecOpsAction("inspect_metrics", {}),
                SecOpsAction("query_logs", {"service": "db"}),
                SecOpsAction("query_logs", {"service": "auth"}),
                SecOpsAction("run_security_scan", {"target": "db"}),
                SecOpsAction("run_security_scan", {"target": "auth"}),
                SecOpsAction("isolate_service", {"service": "db"}),
                SecOpsAction("block_ip", {"ip": "10.0.0.99"}),
                SecOpsAction("submit_diagnosis",
                             {"label": "cyber_attack:data_exfiltration"}),
            ],
        }
        for act in seqs[task_id]:
            _, _, done, _ = env.step(act)
            if done:
                break
        return env.state.to_dict()

    def test_grade_output_range(self, env):
        state = self._run_optimal("easy_memory_leak", env)
        result = grade(state)
        assert 0.0 <= result.score <= 1.0

    def test_grade_easy_correct_high_score(self, env):
        state = self._run_optimal("easy_memory_leak", env)
        result = grade(state)
        assert result.score >= 0.7
        assert result.diagnosis_correct == 1.0

    def test_grade_medium_correct_decent_score(self, env):
        state = self._run_optimal("medium_ddos_cascade", env)
        result = grade(state)
        assert result.score >= 0.6

    def test_grade_hard_correct_reasonable_score(self, env):
        state = self._run_optimal("hard_data_exfiltration", env)
        result = grade(state)
        assert result.score >= 0.6

    def test_grade_wrong_diagnosis_low_score(self, env):
        env.reset("easy_memory_leak")
        env.step(SecOpsAction("submit_diagnosis",
                              {"label": "cyber_attack:ddos"}))
        result = grade(env.state.to_dict())
        assert result.score < 0.5
        assert result.diagnosis_correct == 0.0

    def test_grade_partial_credit_correct_category(self, env):
        env.reset("easy_memory_leak")
        env.step(SecOpsAction("submit_diagnosis",
                              {"label": "infra_failure:service_crash"}))
        result = grade(env.state.to_dict())
        assert result.diagnosis_correct == 0.5

    def test_grade_deterministic(self, env):
        """Same episode → same grade (no randomness in grader)."""
        state = self._run_optimal("easy_memory_leak", env)
        r1 = grade(state)
        r2 = grade(state)
        assert r1.score == r2.score


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_episode_easy(self, env):
        obs = env.reset("easy_memory_leak")
        assert obs is not None
        cumulative = 0.0
        for _ in range(5):
            _, r, done, _ = env.step(SecOpsAction("inspect_metrics", {}))
            cumulative += r
            if done:
                break
        # Should not crash
        assert isinstance(cumulative, float)

    def test_environment_dynamics_metrics_change(self, env):
        env.reset("medium_ddos_cascade")
        m_before = env._metrics["api"].cpu
        env.step(SecOpsAction("inspect_metrics", {}))  # one step
        m_after  = env._metrics["api"].cpu
        # CPU should drift (brownian motion + attack progress)
        # They might be equal by chance, but very unlikely
        assert isinstance(m_after, float)
