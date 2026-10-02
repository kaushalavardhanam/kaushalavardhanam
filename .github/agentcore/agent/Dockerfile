# AgentCore Runtime container for the Claude coding agent.
#
# AgentCore Runtime requires linux/arm64 images. Build with buildx:
#   docker buildx build --platform linux/arm64 -t <ecr-uri> --push .
#
# The container serves POST /invocations + GET /ping on port 8080 (the
# AgentCore Runtime contract); the entrypoint is agent.py.

FROM --platform=linux/arm64 python:3.12-slim

# git for clone/commit/push; certs for HTTPS; ripgrep backs Claude Code's Grep tool.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates ripgrep \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    TEST_VENV_BIN=/opt/testenv/bin

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt requirements-test-env.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Target-repo test dependencies, isolated from the agent's own packages.
RUN python -m venv /opt/testenv \
    && /opt/testenv/bin/pip install --no-cache-dir -r requirements-test-env.txt

# Copy the agent source.
COPY *.py ./

# AgentCore Runtime invokes the container over HTTP on 8080.
EXPOSE 8080

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "agent.py"]
