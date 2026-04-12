"""
Grader for OpenSecOpsEnv
==========================
Deterministic scoring in [0, 1] for each episode.

Score formula:
    score = 0.5 * diagnosis_correct
          + 0.3 * action_efficiency
          + 0.2 * investigation_quality

Where:
    diagnosis_correct  ∈ {0, 1}
    action_efficiency  ∈ [0, 1]  — penalises too many or wrong actions
    investigation_quality ∈ [0, 1] — fraction of relevant services investigated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GradeResult:
    """Detailed grading breakdown for one episode."""

    score: float                  # ∈ [0, 1]  final composite score
    diagnosis_correct: float      # ∈ {0, 1}
    action_efficiency: float      # ∈ [0, 1]
    investigation_quality: float  # ∈ [0, 1]
    details: dict[str, Any]

    def __str__(self) -> str:
        return (
            f"Score: {self.score:.3f} | "
            f"Diagnosis: {self.diagnosis_correct:.1f} | "
            f"Efficiency: {self.action_efficiency:.3f} | "
            f"Investigation: {self.investigation_quality:.3f}"
        )


def grade(episode_state: dict[str, Any]) -> GradeResult:
    """
    Grade a completed episode.

    Parameters
    ----------
    episode_state : dict
        Output of env.state.to_dict() after the episode ends.

    Returns
    -------
    GradeResult
    """
    hidden = episode_state.get("hidden_state", {})
    correct_label = hidden.get("correct_label", "")
    submitted_label = hidden.get("submitted_label", "")

    investigation_actions = episode_state.get("investigation_actions", [])
    mitigation_actions   = episode_state.get("mitigation_actions", [])
    correct_mitigations  = episode_state.get("correct_mitigations", [])
    step_count           = episode_state.get("step_count", 1)
    max_steps            = episode_state.get("max_steps", 50)
    affected_services    = set(hidden.get("affected_services", []))

    # ------------------------------------------------------------------
    # 1.  Diagnosis correctness
    # ------------------------------------------------------------------
    diagnosis_correct = 1.0 if submitted_label == correct_label else 0.0

    # Partial credit: correct root-cause category even if subtype wrong
    if diagnosis_correct == 0.0 and correct_label:
        correct_category = correct_label.split(":")[0]
        submitted_category = submitted_label.split(":")[0] if submitted_label else ""
        if correct_category == submitted_category:
            diagnosis_correct = 0.5

    # ------------------------------------------------------------------
    # 2.  Action efficiency
    # ------------------------------------------------------------------
    # run_security_scan actions land in investigation_actions (not mitigation_actions)
    # because the env classifies them as investigation.  However, the HARD task lists
    # them in correct_mitigations, so we must check BOTH lists.
    all_taken = set(mitigation_actions) | set(investigation_actions)
    n_correct = len(all_taken & set(correct_mitigations))
    n_total_correct = len(correct_mitigations)

    # Fraction of correct mitigations achieved
    mitigation_recall = n_correct / max(n_total_correct, 1)

    # Penalty for wasted steps
    ideal_steps = max(n_total_correct * 3, 5)       # rough ideal budget
    step_ratio  = ideal_steps / max(step_count, 1)
    step_bonus  = min(1.0, step_ratio)               # >1 if faster than ideal

    action_efficiency = round(
        0.7 * mitigation_recall + 0.3 * step_bonus, 4
    )

    # ------------------------------------------------------------------
    # 3.  Investigation quality
    # ------------------------------------------------------------------
    # Which affected services did the agent actually investigate?
    investigated = set()
    for act in investigation_actions:
        # Extract service name from strings like "query_logs:auth"
        parts = act.split(":", 1)
        if len(parts) == 2:
            svc = parts[1]
            if svc in affected_services:
                investigated.add(svc)

    investigation_quality = (
        len(investigated) / len(affected_services)
        if affected_services else 0.0
    )

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------
    score = (
        0.5 * diagnosis_correct
        + 0.3 * action_efficiency
        + 0.2 * investigation_quality
    )
    score = round(max(0.01, min(0.99, score)), 4)

    details = {
        "submitted_label": submitted_label,
        "correct_label": correct_label,
        "diagnosis_correct": diagnosis_correct,
        "n_correct_mitigations": n_correct,
        "n_total_correct_mitigations": n_total_correct,
        "mitigation_recall": mitigation_recall,
        "step_count": step_count,
        "max_steps": max_steps,
        "step_bonus": step_bonus,
        "investigated_affected_services": sorted(investigated),
        "all_affected_services": sorted(affected_services),
    }

    return GradeResult(
        score=score,
        diagnosis_correct=diagnosis_correct,
        action_efficiency=action_efficiency,
        investigation_quality=investigation_quality,
        details=details,
    )
