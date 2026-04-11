"""
server/app.py — Root-level server entry point for OpenSecOpsEnv.

The openenv validate tool expects to find the server at server/app.py
with a main() function that starts the uvicorn server.
"""

import uvicorn

from opensecops_env.server.app import app  # noqa: F401

__all__ = ["app", "main"]


def main() -> None:
    """Start the OpenSecOpsEnv FastAPI server."""
    uvicorn.run(
        "opensecops_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )


if __name__ == "__main__":
    main()
