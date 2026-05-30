"""Shared native Anthropic SSE thinking policy, block remapping, and overlap repair.

Used by :class:`OpenRouterProvider` and line-mode
:class:`providers.anthropic_messages.AnthropicMessagesTransport` providers.

VRL-44 (void-fcc overlay): DeepSeek's ``/anthropic`` endpoint occasionally leaks
its internal tool-call serialization — *DeepSeek Markup Language* (DSML) — into
the user-facing ``text`` content block instead of emitting a proper
``server_tool_use`` / ``tool_use`` block. This happens when the model wants to
issue a *follow-up* tool call after a server-side web_search: the second
invocation surfaces as raw markup like::

    <｜｜DSML｜｜tool_calls>
    <｜｜DSML｜｜invoke name="web_search">
    <｜｜DSML｜｜parameter name="query" string="true">...</｜｜DSML｜｜parameter>
    </｜｜DSML｜｜invoke>
    </｜｜DSML｜｜tool_calls>

That markup belongs to the tool-call channel, not the text the user reads. This
overlay adds a streaming filter that strips complete DSML tool-call spans out of
``text_delta`` content before it is re-emitted downstream, holding back any
partial trailing delimiter across SSE chunks. The filter is a no-op for any text
that contains no DSML markers, so non-DeepSeek native providers are unaffected.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NativeSseBlockPolicyState",
    "format_native_sse_event",
    "is_terminal_openrouter_done_event",
    "parse_native_sse_event",
    "transform_native_sse_block_event",
]

# --- VRL-44: DeepSeek DSML tool-call leak stripper -------------------------------
#
# DSML uses the fullwidth vertical line U+FF5C ("｜") as its delimiter glyph. A
# leaked tool-call span is bounded by an opening ``<｜｜DSML｜｜tool_calls>`` token
# and a closing ``</｜｜DSML｜｜tool_calls>`` token. We strip the whole span
# (including any nested ``<｜｜DSML｜｜invoke ...>`` / ``parameter`` tags).

_DSML_BAR = "｜"  # ｜ U+FF5C FULLWIDTH VERTICAL LINE
_DSML_OPEN = f"<{_DSML_BAR}{_DSML_BAR}DSML{_DSML_BAR}{_DSML_BAR}tool_calls>"
_DSML_CLOSE = f"</{_DSML_BAR}{_DSML_BAR}DSML{_DSML_BAR}{_DSML_BAR}tool_calls>"


def _strip_dsml_tool_calls(text: str, carry: str) -> tuple[str, str]:
    """Strip leaked DSML ``tool_calls`` spans from streamed ``text_delta`` content.

    ``carry`` is text held back from a previous delta (an incomplete DSML marker,
    or text accumulated while inside an unterminated DSML span). Returns
    ``(clean_emit, new_carry)`` where ``clean_emit`` is safe to forward downstream
    immediately and ``new_carry`` must be prepended to the next delta.

    Behaviour:
    - Outside a DSML span: forward text verbatim, but hold back any trailing run
      that could be the start of an ``<｜...`` open marker until we know it isn't.
    - Inside a DSML span (after ``<｜｜DSML｜｜tool_calls>``): drop everything up to
      and including the matching ``</｜｜DSML｜｜tool_calls>``.
    """
    buf = carry + text
    out: list[str] = []

    while buf:
        if buf.startswith("\x00IN_DSML\x00"):
            # Sentinel: we are inside an open DSML span. Look for the close tag.
            inner = buf[len("\x00IN_DSML\x00"):]
            close_at = inner.find(_DSML_CLOSE)
            if close_at == -1:
                # Span not yet closed. If a partial close marker is dangling at the
                # end, keep the whole remainder buffered; otherwise drop the body
                # but retain the sentinel + any trailing partial token.
                keep_from = _dsml_safe_tail(inner)
                return "".join(out), "\x00IN_DSML\x00" + inner[keep_from:]
            # Found the close — drop body + close tag, resume outside the span.
            buf = inner[close_at + len(_DSML_CLOSE):]
            continue

        open_at = buf.find(_DSML_OPEN)
        if open_at == -1:
            # No complete open marker. Emit everything except a trailing run that
            # might be the start of a DSML open token in the next chunk.
            keep_from = _dsml_safe_tail(buf)
            out.append(buf[:keep_from])
            return "".join(out), buf[keep_from:]
        # Emit text before the open marker, then enter the span.
        out.append(buf[:open_at])
        buf = "\x00IN_DSML\x00" + buf[open_at + len(_DSML_OPEN):]

    return "".join(out), ""


def _dsml_safe_tail(s: str) -> int:
    """Index from which the trailing run of ``s`` might begin a DSML marker.

    Returns ``len(s)`` when no suffix of ``s`` is a prefix of a DSML open/close
    token (so all of ``s`` is safe to emit). Otherwise returns the start index of
    the longest such suffix, which must be buffered for the next chunk.
    """
    n = len(s)
    # Longest possible partial marker we might hold is the close token length.
    max_marker = max(len(_DSML_OPEN), len(_DSML_CLOSE))
    start = max(0, n - max_marker)
    for i in range(start, n):
        suffix = s[i:]
        if _DSML_OPEN.startswith(suffix) or _DSML_CLOSE.startswith(suffix):
            return i
    return n


@dataclass
class _UpstreamBlockState:
    """Per-upstream content block: segment index and liveness in the model stream."""

    block_type: str
    down_index: int
    open: bool
    last_start_block: dict[str, Any] | None = None


@dataclass
class NativeSseBlockPolicyState:
    """Track per-upstream content blocks and remapped Anthropic ``index`` field."""

    next_index: int = 0
    by_upstream: dict[int, _UpstreamBlockState] = field(default_factory=dict)
    dropped_indexes: set[int] = field(default_factory=set)
    pending_suppressed_stops: set[int] = field(default_factory=set)
    message_stopped: bool = False
    # VRL-44: per-upstream-index carry buffer for the streaming DSML stripper.
    # Holds text withheld from a text_delta (partial DSML marker or in-span body)
    # that must be reconciled against the next delta for the same block.
    dsml_text_carry: dict[int, str] = field(default_factory=dict)
    # VRL-52: tool-name memory. Captures (id, name) from EVERY tool_use /
    # server_tool_use content_block_start we observe — including starts that
    # fail the int-index guard and never become a segment — so an orphan
    # input_json_delta (no prior recorded start) can recover the real name
    # instead of opening a nameless, client-unparseable tool_use block.
    tool_name_hints: dict[int, dict[str, Any]] = field(default_factory=dict)
    last_tool_name_hint: dict[str, Any] | None = None


def format_native_sse_event(event_name: str | None, data_text: str) -> str:
    """Format an SSE event from its event name and data payload."""
    lines: list[str] = []
    if event_name:
        lines.append(f"event: {event_name}")
    lines.extend(f"data: {line}" for line in data_text.splitlines())
    return "\n".join(lines) + "\n\n"


def parse_native_sse_event(event: str) -> tuple[str | None, str]:
    """Extract the event name and raw data payload from an SSE event."""
    event_name = None
    data_lines: list[str] = []
    for line in event.strip().splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return event_name, "\n".join(data_lines)


def is_terminal_openrouter_done_event(event_name: str | None, data_text: str) -> bool:
    """Return whether an event is OpenAI-style terminal noise (``[DONE]``)."""
    return (event_name is None or event_name in {"data", "done"}) and (
        data_text.strip().upper() == "[DONE]"
    )


def _delta_type_to_block_kind(delta_type: Any) -> str | None:
    """Map a content_block_delta type to a content block kind (text/thinking/tool_use)."""
    if not isinstance(delta_type, str):
        return None
    if delta_type in {"thinking_delta", "signature_delta"}:
        return "thinking"
    if delta_type == "text_delta":
        return "text"
    if delta_type == "input_json_delta":
        return "tool_use"
    return None


def _record_tool_name_hint(state: NativeSseBlockPolicyState, block: Any, index: Any) -> None:
    """Remember a tool_use/server_tool_use start's id+name for later recovery.

    Captures from any observed start (including ones that fail the int-index
    guard) so an orphan input_json_delta can be given the real name. Keyed by
    int index when available; also kept as `last_tool_name_hint` (DeepSeek emits
    tool calls sequentially, so the most recent dangling start is the best
    fallback for an un-keyed orphan delta).
    """
    if not isinstance(block, dict):
        return
    if block.get("type") not in ("tool_use", "server_tool_use"):
        return
    name = block.get("name")
    if not (isinstance(name, str) and name.strip()):
        return
    tool_id = block.get("id")
    hint = {
        "id": tool_id if isinstance(tool_id, str) and tool_id else None,
        "name": name,
    }
    if isinstance(index, int):
        state.tool_name_hints[index] = hint
    state.last_tool_name_hint = hint


def _synthetic_start_content_block(
    block_kind: str,
    *,
    upstream_index: int,
    stored_tool_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a `content_block` for a `content_block_start` with empty streaming fields."""
    if block_kind == "tool_use":
        if (
            isinstance(stored_tool_block, dict)
            and stored_tool_block.get("type") == "tool_use"
        ):
            tool_id = stored_tool_block.get("id")
            name = stored_tool_block.get("name")
            inp = stored_tool_block.get("input")
            return {
                "type": "tool_use",
                "id": tool_id
                if isinstance(tool_id, str) and tool_id
                else f"toolu_or_{upstream_index}",
                "name": name if isinstance(name, str) else "",
                "input": inp if isinstance(inp, dict) else {},
            }
        return {
            "type": "tool_use",
            "id": f"toolu_or_{upstream_index}",
            "name": "",
            "input": {},
        }
    if block_kind == "thinking":
        return {"type": "thinking", "thinking": ""}
    if block_kind == "text":
        return {"type": "text", "text": ""}
    return {"type": "text", "text": ""}


def _should_drop_block_type(block_type: Any, *, thinking_enabled: bool) -> bool:
    if not isinstance(block_type, str):
        return False
    if block_type.startswith("redacted_thinking"):
        return not thinking_enabled
    return not thinking_enabled and "thinking" in block_type


def _synthetic_close_other_open_blocks(
    state: NativeSseBlockPolicyState, current_upstream: int
) -> str:
    """Close every open block except `current_upstream` and track duplicate upstream stops."""
    out: list[str] = []
    for upstream, seg in list(state.by_upstream.items()):
        if upstream == current_upstream or not seg.open:
            continue
        out.append(
            format_native_sse_event(
                "content_block_stop",
                json.dumps(
                    {
                        "type": "content_block_stop",
                        "index": seg.down_index,
                    }
                ),
            )
        )
        seg.open = False
        state.pending_suppressed_stops.add(upstream)
    return "".join(out)


def _allocate_new_segment(
    state: NativeSseBlockPolicyState,
    upstream_index: int,
    block_type: str,
    *,
    last_start_block: dict[str, Any] | None = None,
) -> int:
    """Assign a new downstream `index` for a segment and record upstream state."""
    new_idx = state.next_index
    state.next_index += 1
    state.by_upstream[upstream_index] = _UpstreamBlockState(
        block_type=block_type,
        down_index=new_idx,
        open=True,
        last_start_block=last_start_block,
    )
    return new_idx


def transform_native_sse_block_event(
    event: str,
    state: NativeSseBlockPolicyState,
    *,
    thinking_enabled: bool,
) -> str | None:
    """Normalize native Anthropic SSE events and enforce local thinking policy."""
    event_name, data_text = parse_native_sse_event(event)
    if not event_name or not data_text:
        return event

    try:
        payload = json.loads(data_text)
    except json.JSONDecodeError:
        return event

    if event_name == "content_block_start":
        block = payload.get("content_block")
        if not isinstance(block, dict):
            return event
        block_type = block.get("type")
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            _record_tool_name_hint(state, block, upstream_index)
            return event
        if _should_drop_block_type(block_type, thinking_enabled=thinking_enabled):
            state.dropped_indexes.add(upstream_index)
            return None

        if not isinstance(block_type, str):
            return event
        prefix = _synthetic_close_other_open_blocks(state, upstream_index)
        stored = copy.deepcopy(block)
        _record_tool_name_hint(state, block, upstream_index)
        new_idx = _allocate_new_segment(
            state,
            upstream_index,
            block_type=block_type,
            last_start_block=stored,
        )
        payload["index"] = new_idx
        return prefix + format_native_sse_event(event_name, json.dumps(payload))

    if event_name == "content_block_delta":
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return event
        delta_type = delta.get("type")
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return event
        if upstream_index in state.dropped_indexes:
            return None
        if _should_drop_block_type(delta_type, thinking_enabled=thinking_enabled):
            return None

        block_kind = _delta_type_to_block_kind(delta_type)
        if block_kind is None:
            return event

        # VRL-44: strip leaked DeepSeek DSML tool-call markup from text deltas.
        # The markup belongs to the tool-call channel, not the user-facing text.
        # Streaming-aware: text inside an (unterminated) DSML span is buffered in
        # `state.dsml_text_carry[upstream_index]` and reconciled with later deltas.
        if delta_type == "text_delta" and isinstance(delta.get("text"), str):
            carry = state.dsml_text_carry.get(upstream_index, "")
            clean, new_carry = _strip_dsml_tool_calls(delta["text"], carry)
            state.dsml_text_carry[upstream_index] = new_carry
            if not clean:
                # Entire delta was DSML markup (or fully buffered) — drop it.
                return None
            delta["text"] = clean

        seg = state.by_upstream.get(upstream_index)
        if seg and seg.open:
            payload["index"] = seg.down_index
            return format_native_sse_event(event_name, json.dumps(payload))

        if seg is not None and not seg.open:
            # More deltas for an upstream block after a synthetic (or other) close:
            # reopen with a new downstream `index` and emit a synthetic `content_block_start` first.
            state.pending_suppressed_stops.discard(upstream_index)
            carry = seg.last_start_block
            new_idx = _allocate_new_segment(
                state,
                upstream_index,
                block_type=block_kind,
                last_start_block=carry,
            )
            stored_tool = (
                carry
                if isinstance(carry, dict) and carry.get("type") == "tool_use"
                else None
            )
            start_payload = {
                "type": "content_block_start",
                "index": new_idx,
                "content_block": _synthetic_start_content_block(
                    block_kind,
                    upstream_index=upstream_index,
                    stored_tool_block=stored_tool,
                ),
            }
            prefix = format_native_sse_event(
                "content_block_start", json.dumps(start_payload)
            )
            payload["index"] = new_idx
            return prefix + format_native_sse_event(event_name, json.dumps(payload))

        # Delta with no prior `content_block_start` in this stream
        if block_kind == "text":
            synthetic_block = _synthetic_start_content_block(
                block_kind,
                upstream_index=upstream_index,
            )
        elif block_kind == "tool_use":
            hint = state.tool_name_hints.get(upstream_index) or state.last_tool_name_hint
            if not (isinstance(hint, dict) and isinstance(hint.get("name"), str) and hint["name"].strip()):
                # Unrecoverable: do NOT open a nameless tool_use block. Suppress
                # the orphan delta. Losing one un-nameable tool call (agent
                # retries) beats poisoning the stream with an unparseable block.
                print(
                    f"[VRL-52] suppressing orphan input_json_delta at index "
                    f"{upstream_index}: no recoverable tool name",
                    file=sys.stderr,
                )
                return None
            stored_tool_block = {
                "type": "tool_use",
                "id": hint.get("id") or f"toolu_or_{upstream_index}",
                "name": hint["name"],
                "input": {},
            }
            synthetic_block = _synthetic_start_content_block(
                block_kind,
                upstream_index=upstream_index,
                stored_tool_block=stored_tool_block,
            )
        else:
            # thinking: pass through raw (unusual upstream shape)
            return event
        new_idx = _allocate_new_segment(
            state,
            upstream_index,
            block_type=block_kind,
            last_start_block=copy.deepcopy(synthetic_block),
        )
        start_payload = {
            "type": "content_block_start",
            "index": new_idx,
            "content_block": synthetic_block,
        }
        prefix = format_native_sse_event(
            "content_block_start", json.dumps(start_payload)
        )
        payload["index"] = new_idx
        return prefix + format_native_sse_event(event_name, json.dumps(payload))

    if event_name == "content_block_stop":
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return event
        if upstream_index in state.dropped_indexes:
            return None
        if upstream_index in state.pending_suppressed_stops:
            state.pending_suppressed_stops.discard(upstream_index)
            return None

        # VRL-44: flush any carry the DSML stripper held back for this block.
        # Benign trailing text (a partial open-marker that never completed) is
        # emitted as a final text_delta; text still inside an unterminated DSML
        # span (sentinel-prefixed) is dropped — it is tool-call markup.
        flush_prefix = ""
        carry = state.dsml_text_carry.pop(upstream_index, "")
        if carry and not carry.startswith("\x00IN_DSML\x00"):
            seg_for_flush = state.by_upstream.get(upstream_index)
            if seg_for_flush is not None and seg_for_flush.open:
                flush_prefix = format_native_sse_event(
                    "content_block_delta",
                    json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": seg_for_flush.down_index,
                            "delta": {"type": "text_delta", "text": carry},
                        }
                    ),
                )

        seg = state.by_upstream.get(upstream_index)
        if seg is not None and seg.open:
            payload["index"] = seg.down_index
            seg.open = False
            return flush_prefix + format_native_sse_event(
                event_name, json.dumps(payload)
            )
        if seg is not None:
            # Spurious or duplicate `content_block_stop` for a closed block.
            return None
        if not thinking_enabled:
            return None
        return event

    return event
