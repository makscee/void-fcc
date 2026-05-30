"""VRL-52: Regression tests — no tool_use content_block_start with name=="".

Covers:
1. NativeSseBlockPolicyState has tool_name_hints field.
2. Guard-failing starts (non-int index) still record last_tool_name_hint.
3. Normal starts record tool_name_hints by int index.
4. Orphan input_json_delta at a guard-failed start index recovers the real name.
5. Orphan input_json_delta with no prior hint at all is suppressed (returns None).
6. Mixed stream never emits a tool_use start with name == "".
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PATCH_FILE = (
    Path(__file__).resolve().parents[1] / "patches" / "native_sse_block_policy_vrl44.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("native_sse_block_policy_vrl44", PATCH_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()
NativeSseBlockPolicyState = MOD.NativeSseBlockPolicyState
transform_native_sse_block_event = MOD.transform_native_sse_block_event
format_native_sse_event = MOD.format_native_sse_event


def _tool_start(idx, *, name="get_weather", tool_id="toolu_abc"):
    return format_native_sse_event(
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
        }),
    )


def _json_delta(idx, partial='{"city":'):
    return format_native_sse_event(
        "content_block_delta",
        json.dumps({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        }),
    )


def _starts_emitted(events, st):
    """Run events; return list of tool_use content_block dicts emitted as starts."""
    blocks = []
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
                    if payload.get("type") == "content_block_start":
                        cb = payload.get("content_block")
                        if isinstance(cb, dict) and cb.get("type") == "tool_use":
                            blocks.append(cb)
    return blocks


# --- Task 1: state field exists ---

def test_state_has_tool_name_hint_field():
    st = NativeSseBlockPolicyState()
    assert hasattr(st, "tool_name_hints")
    assert st.tool_name_hints == {}
    assert hasattr(st, "last_tool_name_hint")
    assert st.last_tool_name_hint is None


# --- Task 2: capture at guard + normal start ---

def test_guard_failing_start_still_records_name():
    st = NativeSseBlockPolicyState()
    bad = format_native_sse_event(
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": "0",  # string, not int -> hits the guard early-return
            "content_block": {"type": "tool_use", "id": "toolu_x", "name": "search_web", "input": {}},
        }),
    )
    out = transform_native_sse_block_event(bad, st, thinking_enabled=True)
    assert out is not None  # raw event still passed through unchanged
    assert st.last_tool_name_hint == {"id": "toolu_x", "name": "search_web"}


def test_normal_start_records_name_by_index():
    st = NativeSseBlockPolicyState()
    transform_native_sse_block_event(_tool_start(0, name="get_weather", tool_id="toolu_w"), st, thinking_enabled=True)
    assert st.tool_name_hints.get(0) == {"id": "toolu_w", "name": "get_weather"}


# --- Task 3: no-prior-start branch recovers real name ---

def test_orphan_delta_recovers_name_from_guard_failed_start():
    st = NativeSseBlockPolicyState()
    bad_start = format_native_sse_event(
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": "0",  # guard-failing
            "content_block": {"type": "tool_use", "id": "toolu_real", "name": "search_web", "input": {}},
        }),
    )
    events = [bad_start, _json_delta(0, '{"q":')]
    starts = _starts_emitted(events, st)
    tool_starts = [b for b in starts if b.get("type") == "tool_use"]
    assert tool_starts, "expected a synthetic tool_use start to be emitted"
    assert all(b.get("name") for b in tool_starts), "no tool_use start may have empty name"
    assert tool_starts[-1]["name"] == "search_web"
    assert tool_starts[-1]["id"] == "toolu_real"


def test_no_tool_use_start_ever_has_empty_name():
    st = NativeSseBlockPolicyState()
    # Orphan delta with a recoverable hint present (guard-failing start).
    transform_native_sse_block_event(
        format_native_sse_event("content_block_start", json.dumps({
            "type": "content_block_start", "index": "1",
            "content_block": {"type": "tool_use", "id": "toolu_z", "name": "calc", "input": {}},
        })), st, thinking_enabled=True,
    )
    starts = _starts_emitted([_json_delta(1)], st)
    assert all(b.get("name", "") != "" for b in starts)


# --- Task 4: unrecoverable case — suppress, don't emit nameless ---

def test_orphan_delta_with_no_hint_is_suppressed():
    st = NativeSseBlockPolicyState()
    out = transform_native_sse_block_event(_json_delta(0, '{"x":1}'), st, thinking_enabled=True)
    assert out is None  # suppressed entirely, no nameless block opened


def test_orphan_delta_no_hint_emits_no_tool_use_start():
    st = NativeSseBlockPolicyState()
    starts = _starts_emitted([_json_delta(0), _json_delta(0, '2}')], st)
    assert starts == []  # never opened a segment, so nothing to reopen either


def test_mixed_stream_never_emits_empty_name_tool_use():
    st = NativeSseBlockPolicyState()
    events = [
        _tool_start(0, name="get_weather", tool_id="toolu_a"),
        _json_delta(0, '{"city":"NYC"}'),
        format_native_sse_event("content_block_start", json.dumps({
            "type": "content_block_start", "index": "1",
            "content_block": {"type": "tool_use", "id": "toolu_b", "name": "search", "input": {}},
        })),
        _json_delta(1, '{"q":"x"}'),
        _json_delta(2, '{"orphan":true}'),  # no hint -> suppressed
    ]
    starts = _starts_emitted(events, st)
    for b in starts:
        if b.get("type") == "tool_use":
            assert b.get("name", "") != "", f"nameless tool_use emitted: {b}"


# --- Task 5: real-path replay fixture ---

def test_real_path_replay_orphan_delta_round_trip():
    """Replay of a real-shaped DeepSeek SSE stream exhibiting the orphan-delta pattern.

    Scenario: DeepSeek emits a tool_use content_block_start with a non-int index
    (guard-failing), then immediately emits an input_json_delta for int index 0.
    Before VRL-52, _synthetic_start_content_block's fallback produced name=="",
    causing the Claude Code client to render [unsupported block type: tool_use].
    After VRL-52, the real name is recovered from last_tool_name_hint.
    """
    st = NativeSseBlockPolicyState()

    # Step 1: guard-failing start carries real name but is not recorded as segment
    guard_failing_start = format_native_sse_event(
        "content_block_start",
        json.dumps({
            "type": "content_block_start",
            "index": "0",  # non-int: hits int-index guard, passes through raw
            "content_block": {
                "type": "tool_use",
                "id": "toolu_ds_001",
                "name": "web_search",
                "input": {},
            },
        }),
    )

    # Step 2: DeepSeek emits input_json_delta for int index 0 (orphan: no segment recorded)
    orphan_delta = _json_delta(0, '{"query": "latest news"}')

    # Step 3: more input payload
    cont_delta = _json_delta(0, '"latest news"}')

    events = [guard_failing_start, orphan_delta, cont_delta]
    starts = _starts_emitted(events, st)

    tool_starts = [b for b in starts if b.get("type") == "tool_use"]
    assert tool_starts, "must emit at least one tool_use start"

    # Invariant: no empty name (the VRL-52 regression guard)
    for b in tool_starts:
        assert b.get("name", "") != "", (
            "VRL-52 regression: tool_use start emitted with empty name; "
            "before the fix this branch returned name=='' from _synthetic_start_content_block fallback"
        )

    # Positive: the recovered name and id match the guard-failing start
    final = tool_starts[-1]
    assert final["name"] == "web_search"
    assert final["id"] == "toolu_ds_001"
