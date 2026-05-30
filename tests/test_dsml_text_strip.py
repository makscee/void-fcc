"""VRL-44 — strip leaked DeepSeek DSML tool-call markup from text blocks.

DeepSeek's ``/anthropic`` endpoint sometimes leaks its internal tool-call
serialization (DSML — DeepSeek Markup Language) into the user-facing ``text``
content block when it wants to issue a follow-up tool call after a server-side
web_search. The leaked markup looks like::

    <｜｜DSML｜｜tool_calls>
    <｜｜DSML｜｜invoke name="web_search">
    <｜｜DSML｜｜parameter name="query" string="true">...</｜｜DSML｜｜parameter>
    </｜｜DSML｜｜invoke>
    </｜｜DSML｜｜tool_calls>

The overlay patch ``patches/native_sse_block_policy_vrl44.py`` adds a streaming
filter that strips complete DSML tool-call spans out of ``text_delta`` content
before it is forwarded downstream, holding back partial markers across SSE
chunks. The patch file has no third-party imports (stdlib only) so it loads by
path directly.

RED (before the stripper logic): leaked DSML markup reaches the assistant text.
GREEN (after): only the real answer text survives; web_search blocks untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PATCH_FILE = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "native_sse_block_policy_vrl44.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "native_sse_block_policy_vrl44", PATCH_FILE
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass introspection (which looks the module up
    # in sys.modules via cls.__module__) works on Python 3.14.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()
NativeSseBlockPolicyState = MOD.NativeSseBlockPolicyState
transform_native_sse_block_event = MOD.transform_native_sse_block_event
format_native_sse_event = MOD.format_native_sse_event

BAR = "｜"  # U+FF5C fullwidth vertical line
DSML_OPEN = f"<{BAR}{BAR}DSML{BAR}{BAR}tool_calls>"
DSML_CLOSE = f"</{BAR}{BAR}DSML{BAR}{BAR}tool_calls>"

# The exact leaked sample captured from DeepSeek /anthropic (VRL-44 repro).
LEAKED = (
    f"{DSML_OPEN}\n"
    f"<{BAR}{BAR}DSML{BAR}{BAR}invoke name=\"web_search\">\n"
    f"<{BAR}{BAR}DSML{BAR}{BAR}parameter name=\"query\" string=\"true\">"
    f"John Hopfield 1982 paper key findings"
    f"</{BAR}{BAR}DSML{BAR}{BAR}parameter>\n"
    f"</{BAR}{BAR}DSML{BAR}{BAR}invoke>\n"
    f"{DSML_CLOSE}"
)


def _text_start(idx: int) -> str:
    return format_native_sse_event(
        "content_block_start",
        json.dumps(
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            }
        ),
    )


def _text_delta(idx: int, text: str) -> str:
    return format_native_sse_event(
        "content_block_delta",
        json.dumps(
            {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": text},
            }
        ),
    )


def _text_stop(idx: int) -> str:
    return format_native_sse_event(
        "content_block_stop",
        json.dumps({"type": "content_block_stop", "index": idx}),
    )


def _emitted_text(events: list[str], st: NativeSseBlockPolicyState) -> str:
    """Run events through the transform and collect emitted text_delta content."""
    out_text: list[str] = []
    for ev in events:
        result = transform_native_sse_block_event(ev, st, thinking_enabled=True)
        if not result:
            continue
        for raw in result.split("\n\n"):
            for line in raw.splitlines():
                if line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    delta = payload.get("delta")
                    if (
                        isinstance(delta, dict)
                        and delta.get("type") == "text_delta"
                    ):
                        out_text.append(delta.get("text", ""))
    return "".join(out_text)


# --------------------------------------------------------------------------- #
# RED-anchor tests: the leak must be gone.                                     #
# --------------------------------------------------------------------------- #


def test_full_dsml_span_in_one_delta_is_stripped() -> None:
    st = NativeSseBlockPolicyState()
    events = [_text_start(0), _text_delta(0, LEAKED), _text_stop(0)]
    assert _emitted_text(events, st) == ""


def test_real_answer_with_trailing_dsml_keeps_answer_drops_markup() -> None:
    st = NativeSseBlockPolicyState()
    answer = "Tokyo's 2025 population is about 37 million.\n\n"
    events = [
        _text_start(0),
        _text_delta(0, answer),
        _text_delta(0, LEAKED),
        _text_stop(0),
    ]
    emitted = _emitted_text(events, st)
    assert "DSML" not in emitted
    assert "tool_calls" not in emitted
    assert emitted == answer


def test_dsml_split_across_many_deltas_is_stripped() -> None:
    st = NativeSseBlockPolicyState()
    # Chunk the leaked markup into 7-char pieces to simulate SSE token streaming,
    # including splitting right inside the fullwidth-bar delimiters.
    chunks = [LEAKED[i : i + 7] for i in range(0, len(LEAKED), 7)]
    events = [_text_start(0)] + [_text_delta(0, c) for c in chunks] + [_text_stop(0)]
    emitted = _emitted_text(events, st)
    assert emitted == ""
    assert "DSML" not in emitted


def test_dsml_between_two_real_text_runs() -> None:
    st = NativeSseBlockPolicyState()
    events = [
        _text_start(0),
        _text_delta(0, "Before. "),
        _text_delta(0, LEAKED),
        _text_delta(0, "After."),
        _text_stop(0),
    ]
    emitted = _emitted_text(events, st)
    assert emitted == "Before. After."
    assert "DSML" not in emitted


# --------------------------------------------------------------------------- #
# Guard tests: clean paths and non-text channels are untouched.               #
# --------------------------------------------------------------------------- #


def test_clean_text_passes_through_unchanged() -> None:
    st = NativeSseBlockPolicyState()
    answer = "The population of Tokyo in 2025 is approximately 37 million."
    events = [_text_start(0), _text_delta(0, answer), _text_stop(0)]
    assert _emitted_text(events, st) == answer


def test_benign_text_with_lt_bar_not_dsml_is_preserved() -> None:
    st = NativeSseBlockPolicyState()
    # Contains the fullwidth bar but is NOT a DSML token — must survive intact,
    # including across the end-of-stream flush.
    answer = f"Math: a < b and the pipe glyph {BAR} is harmless."
    events = [_text_start(0), _text_delta(0, answer), _text_stop(0)]
    assert _emitted_text(events, st) == answer


def test_server_tool_use_block_is_not_affected() -> None:
    st = NativeSseBlockPolicyState()
    start = format_native_sse_event(
        "content_block_start",
        json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {},
                },
            }
        ),
    )
    out = transform_native_sse_block_event(start, st, thinking_enabled=True)
    assert out is not None
    assert "server_tool_use" in out
    assert "web_search" in out


def test_web_search_tool_result_block_is_not_affected() -> None:
    st = NativeSseBlockPolicyState()
    start = format_native_sse_event(
        "content_block_start",
        json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [],
                },
            }
        ),
    )
    out = transform_native_sse_block_event(start, st, thinking_enabled=True)
    assert out is not None
    assert "web_search_tool_result" in out
