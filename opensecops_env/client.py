"""
OpenSecOpsEnv – Client
=======================
HTTP client that connects to a running OpenSecOpsEnv server.
Follows the OpenEnv EnvClient pattern.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from opensecops_env.models import (
    SecOpsAction,
    SecOpsObservation,
    SecOpsState,
)


class SecOpsEnvClient:
    """
    Thin synchronous HTTP client for OpenSecOpsEnv.

    Parameters
    ----------
    base_url : str
        Base URL of the running server, e.g. "http://localhost:8000"

    Example
    -------
    >>> client = SecOpsEnvClient("http://localhost:8000")
    >>> obs = client.reset("easy_memory_leak")
    >>> obs, reward, done, info = client.step(
    ...     SecOpsAction(action_type="query_logs",
    ...                  parameters={"service": "auth"})
    ... )
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy_memory_leak") -> SecOpsObservation:
        data = self._post("/reset", {"task_id": task_id})
        return self._parse_obs(data["observation"])

    def step(
        self, action: SecOpsAction
    ) -> tuple[SecOpsObservation, float, bool, dict[str, Any]]:
        payload = {
            "action_type": action.action_type,
            "parameters": action.parameters,
        }
        data = self._post("/step", payload)
        obs = self._parse_obs(data["observation"])
        return obs, data["reward"], data["done"], data.get("info", {})

    def state(self) -> dict[str, Any]:
        import urllib.request
        with urllib.request.urlopen(f"{self.base_url}/state") as resp:
            return json.loads(resp.read())["state"]

    def grade(self) -> dict[str, Any]:
        return self._post("/grade", {})

    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload).encode()
        if HAS_REQUESTS:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        # stdlib fallback
        import urllib.request
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    @staticmethod
    def _parse_obs(d: dict) -> SecOpsObservation:
        return SecOpsObservation(
            alerts=d.get("alerts", []),
            metrics=d.get("metrics", {}),
            logs=d.get("logs", []),
            topology=d.get("topology", {}),
            last_action_result=d.get("last_action_result", ""),
            time_step=d.get("time_step", 0),
            available_actions=d.get("available_actions", []),
        )
