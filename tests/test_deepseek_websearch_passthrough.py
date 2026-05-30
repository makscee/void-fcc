"""VRL-44 — DeepSeek native web_search pass-through.

DeepSeek's /anthropic endpoint (which fcc's DeepSeek provider already targets)
natively executes web search server-side. The patch must therefore STOP
rejecting the `web_search`/`web_fetch` server-tool definitions and STOP stripping
the `server_tool_use`/`web_search_tool_result`/`web_fetch_tool_result` result
blocks, so they pass through to DeepSeek.

This test loads the patch file directly by path with lightweight stubs for the
free-claude-code upstream modules it imports (config.constants, core.anthropic,
providers.exceptions, loguru). That lets it run in CI / locally without the full
free-claude-code install.

It also guards the VCD-34 stub-thinking behavior, which must remain intact.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PATCH_FILE = Path(__file__).resolve().parents[1] / "patches" / "deepseek_request_vcd34.py"


def _install_upstream_stubs() -> None:
    """Register minimal stand-ins for the free-claude-code modules the patch imports."""
    # loguru.logger — any attribute access returns a no-op callable.
    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _NoopLogger:
            def __getattr__(self, _name):
                def _noop(*_args, **_kwargs):
                    return None

                return _noop

        loguru.logger = _NoopLogger()
        sys.modules["loguru"] = loguru

    # config.constants.ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    if "config" not in sys.modules:
        config_pkg = types.ModuleType("config")
        config_pkg.__path__ = []  # mark as package
        sys.modules["config"] = config_pkg
    if "config.constants" not in sys.modules:
        constants = types.ModuleType("config.constants")
        constants.ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 8192
        sys.modules["config.constants"] = constants
        sys.modules["config"].constants = constants

    # core.anthropic.native_messages_request.dump_raw_messages_request
    if "core" not in sys.modules:
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = []
        sys.modules["core"] = core_pkg
    if "core.anthropic" not in sys.modules:
        core_anthropic = types.ModuleType("core.anthropic")
        core_anthropic.__path__ = []
        sys.modules["core.anthropic"] = core_anthropic
        sys.modules["core"].anthropic = core_anthropic
    if "core.anthropic.native_messages_request" not in sys.modules:
        nmr = types.ModuleType("core.anthropic.native_messages_request")

        def dump_raw_messages_request(request_data):
            # In tests we pass a plain dict already; just deep-ish copy it.
            if isinstance(request_data, dict):
                return dict(request_data)
            return request_data

        nmr.dump_raw_messages_request = dump_raw_messages_request
        sys.modules["core.anthropic.native_messages_request"] = nmr
        sys.modules["core.anthropic"].native_messages_request = nmr

    # providers.exceptions.InvalidRequestError
    if "providers" not in sys.modules:
        providers_pkg = types.ModuleType("providers")
        providers_pkg.__path__ = []
        sys.modules["providers"] = providers_pkg
    if "providers.exceptions" not in sys.modules:
        exc = types.ModuleType("providers.exceptions")

        class InvalidRequestError(Exception):
            pass

        exc.InvalidRequestError = InvalidRequestError
        sys.modules["providers.exceptions"] = exc
        sys.modules["providers"].exceptions = exc


def _load_patch_module():
    _install_upstream_stubs()
    spec = importlib.util.spec_from_file_location("deepseek_request_vcd34", PATCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_patch_module()


# ── web_search tool definition must NOT be rejected ──────────────────────────

def test_web_search_tool_def_passes_validation(mod):
    """A request carrying the web_search server tool must pass through (no raise)."""
    data = {
        "model": "deepseek-v4-pro",
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        "messages": [{"role": "user", "content": "latest python version?"}],
    }
    # Must not raise InvalidRequestError.
    mod._validate_deepseek_native_request_dict(data)


def test_web_fetch_tool_def_passes_validation(mod):
    data = {
        "model": "deepseek-v4-pro",
        "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
        "messages": [{"role": "user", "content": "fetch this page"}],
    }
    mod._validate_deepseek_native_request_dict(data)


# ── web_search result blocks must NOT be stripped/rejected in history ─────────

def test_server_tool_use_block_not_unsupported(mod):
    assert "server_tool_use" not in mod._UNSUPPORTED_MESSAGE_BLOCK_TYPES
    assert "web_search_tool_result" not in mod._UNSUPPORTED_MESSAGE_BLOCK_TYPES
    assert "web_fetch_tool_result" not in mod._UNSUPPORTED_MESSAGE_BLOCK_TYPES


def test_web_search_history_passes_validation(mod):
    """A follow-up turn replaying server_tool_use + web_search_tool_result must pass."""
    data = {
        "model": "deepseek-v4-pro",
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [
            {"role": "user", "content": "latest python version?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "srv_1", "name": "web_search",
                     "input": {"query": "latest python version"}},
                    {"type": "web_search_tool_result", "tool_use_id": "srv_1",
                     "content": [{"type": "web_search_result", "title": "x", "url": "y"}]},
                    {"type": "text", "text": "Python 3.14"},
                ],
            },
        ],
    }
    mod._validate_deepseek_native_request_dict(data)


# ── image/document blocks must STILL be rejected/stripped (unchanged) ─────────

def test_image_block_still_unsupported(mod):
    assert "image" in mod._UNSUPPORTED_MESSAGE_BLOCK_TYPES
    assert "document" in mod._UNSUPPORTED_MESSAGE_BLOCK_TYPES


def test_image_block_still_raises_in_validation(mod):
    from providers.exceptions import InvalidRequestError

    data = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "x"}},
            ]},
        ],
    }
    with pytest.raises(InvalidRequestError):
        mod._validate_deepseek_native_request_dict(data)


def test_mcp_servers_still_rejected(mod):
    from providers.exceptions import InvalidRequestError

    data = {"model": "deepseek-v4-pro", "mcp_servers": [{"url": "x"}], "messages": []}
    with pytest.raises(InvalidRequestError):
        mod._validate_deepseek_native_request_dict(data)


# ── VCD-34 stub-thinking behavior must remain intact ─────────────────────────

def test_stub_thinking_still_injected_for_tool_use(mod):
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
        ]},
    ]
    out = mod._inject_stub_thinking_for_tool_use(messages)
    first_block = out[0]["content"][0]
    assert first_block["type"] == "thinking"
    assert first_block["thinking"] == ""
