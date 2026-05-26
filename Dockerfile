FROM python:3.14-slim

# VRL-19 — void-fcc sidecar: wraps free-claude-code PyPI package as an
# Anthropic-API-compatible HTTP server that proxies to DeepSeek.
# Pinned to free-claude-code==2.0.0 (PyPI). Bump explicitly by PR.

RUN pip install --no-cache-dir free-claude-code==2.0.0

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8082/v1/models', timeout=2)"

# fcc-server is the console_script entrypoint from free-claude-code.
# Env: DEEPSEEK_API_KEY (secret, from SOPS), ANTHROPIC_AUTH_TOKEN (=freecc).
ENTRYPOINT ["fcc-server", "--host", "0.0.0.0", "--port", "8082"]
