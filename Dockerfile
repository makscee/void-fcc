FROM python:3.14-slim

# VRL-19 — void-fcc sidecar: wraps free-claude-code as an
# Anthropic-API-compatible HTTP server that proxies to DeepSeek.
# Pinned to upstream commit 8ae7795961f05f425bd3e7418f85f2ccec7f4600
# (2026-05-24, version = "2.0.0" in pyproject.toml).
# Not on PyPI — install from GitHub source directly.
# Bump: update the @<sha> pin and open a PR review on upstream changes.

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir \
      "git+https://github.com/Alishahryar1/free-claude-code.git@8ae7795961f05f425bd3e7418f85f2ccec7f4600" && \
    apt-get purge -y --auto-remove git && \
    rm -rf /var/lib/apt/lists/*

# VCD-34 r5: patch deepseek/request.py to inject stub thinking blocks for
# tool follow-up turns. deepseek-v4-pro/-flash always emit thinking blocks;
# DeepSeek requires them passed back in subsequent turns. Since claude code
# strips thinking from its context, FCC injects empty stubs to satisfy the
# DeepSeek API requirement.
COPY patches/deepseek_request_vcd34.py \
     /usr/local/lib/python3.14/site-packages/providers/deepseek/request.py

# VRL-44: patch the shared native SSE block policy to strip leaked DeepSeek
# DSML tool-call markup (<｜｜DSML｜｜tool_calls>…) out of the user-facing text
# content block. DeepSeek's /anthropic endpoint sometimes serializes a
# follow-up tool call as raw markup inside a text_delta instead of a proper
# server_tool_use block; this filter removes those spans (streaming-aware, a
# no-op when no DSML markers are present, so other native providers are
# unaffected). web_search pass-through (request.py patch above) is preserved.
COPY patches/native_sse_block_policy_vrl44.py \
     /usr/local/lib/python3.14/site-packages/core/anthropic/native_sse_block_policy.py

EXPOSE 8082

# fcc reads HOST/PORT from env; defaults are 0.0.0.0:8082 in settings.py.
# Set explicitly so they're visible in `docker inspect`.
ENV HOST=0.0.0.0
ENV PORT=8082

# Healthcheck: /health is the server's own preflight endpoint (no auth).
# Falls back to /v1/models with auth token if /health is missing.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8082/health', timeout=2)"

# fcc-server is the console_script entrypoint from free-claude-code.
# Takes no CLI args — all config via env vars (pydantic-settings).
# Env: DEEPSEEK_API_KEY (secret, from SOPS), ANTHROPIC_AUTH_TOKEN (=freecc).
ENTRYPOINT ["fcc-server"]
