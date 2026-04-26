# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ---- OS dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Python dependencies (cached layer) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy project ----
COPY opensecops_env/ /app/opensecops_env/
COPY openenv.yaml       /app/openenv.yaml
COPY pyproject.toml     /app/pyproject.toml
COPY inference.py       /app/inference.py
COPY README.md          /app/README.md
COPY hf_blog_post.md    /app/hf_blog_post.md
COPY training_results.png /app/training_results.png

# ---- Install package (editable so imports resolve correctly) ----
RUN pip install --no-cache-dir -e /app

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---- Expose port ----
EXPOSE 8000

# ---- Run server ----
CMD ["uvicorn", "opensecops_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
