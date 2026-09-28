# AgentCore Runtime container for the decomposer agent.
#
# AgentCore Runtime requires linux/arm64 images. Build with buildx:
#   docker buildx build --platform linux/arm64 -t <ecr-uri> --push .
#
# The container serves POST /invocations + GET /ping on port 8080 (the
# AgentCore Runtime contract); the entrypoint is agent.py.

FROM --platform=linux/arm64 python:3.12-slim

# git is required for the clone/commit/push flow; certs for HTTPS to GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent source.
COPY bedrock_client.py decomposer.py github_app.py git_pr.py agent.py ./

# AgentCore Runtime invokes the container over HTTP on 8080.
EXPOSE 8080

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "agent.py"]
