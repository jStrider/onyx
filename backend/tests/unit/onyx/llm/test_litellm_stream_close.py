"""Deterministic close of litellm sync streams.

An abandoned sync stream is otherwise finalized by the periodic GC, which can
run on a thread holding httpcore's non-reentrant connection-pool lock and
permanently deadlock litellm's shared module-level client. These tests verify
the innermost httpx generator is closed on every non-exhausted exit path.
"""

from collections.abc import Generator, Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.llm.interfaces import LanguageModelInput
from onyx.llm.model_response import Delta, ModelResponseStream, StreamingChoice
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM, _close_litellm_stream
from onyx.tracing.framework.scope import Scope


@pytest.fixture(autouse=True)
def _reset_tracing_scope() -> Iterator[None]:
    # Abandoning a traced stream (GeneratorExit) intentionally leaves the
    # finished span in the tracing contextvar (see SpanImpl.__exit__);
    # production relies on the stale-span guard, but tests need clean state.
    yield
    Scope.set_current_span(None)
    Scope.set_current_trace(None)


class _FakeProviderIterator:
    """Shaped like Anthropic's ModelResponseIterator."""

    def __init__(self, streaming_response: Iterator[bytes]) -> None:
        self.streaming_response = streaming_response


class _FakeStreamWrapper:
    """Shaped like litellm's CustomStreamWrapper (no sync close())."""

    def __init__(
        self,
        completion_stream: Any,
        chunks: list[object],
        error: Exception | None = None,
    ) -> None:
        self.completion_stream = completion_stream
        self._chunks = chunks
        self._error = error

    def __iter__(self) -> Iterator[object]:
        yield from self._chunks
        if self._error is not None:
            raise self._error


def _tracking_generator(closed_flag: list[bool]) -> Iterator[bytes]:
    try:
        while True:
            yield b"data: {}"
    finally:
        closed_flag[0] = True


def _make_fake_llm() -> MagicMock:
    llm = MagicMock()
    llm.config.model_name = "claude-test"
    llm.config.model_provider = "anthropic"
    llm._timeout = 30
    llm._track_llm_cost = MagicMock()
    return llm


def _make_prompt() -> LanguageModelInput:
    return [UserMessage(content="hello")]


def _make_stream_response(content: str) -> ModelResponseStream:
    return ModelResponseStream(
        id="chunk-1",
        created="1",
        choice=StreamingChoice(delta=Delta(content=content)),
    )


def test_close_litellm_stream_closes_inner_generator() -> None:
    closed = [False]
    gen = _tracking_generator(closed)
    next(gen)  # suspend the generator mid-stream, like a partially-read response
    wrapper = _FakeStreamWrapper(_FakeProviderIterator(gen), chunks=[])

    _close_litellm_stream(wrapper)

    assert closed[0] is True


def test_close_litellm_stream_tolerates_uncloseable_shapes() -> None:
    # Mock/cached streams carry a plain list; nothing to close, nothing raised.
    _close_litellm_stream(_FakeStreamWrapper([1, 2, 3], chunks=[]))
    _close_litellm_stream(object())


def test_invoke_closes_stream_on_midstream_error() -> None:
    closed = [False]
    gen = _tracking_generator(closed)
    next(gen)
    wrapper = _FakeStreamWrapper(
        _FakeProviderIterator(gen),
        chunks=[object()],
        error=RuntimeError("provider blew up mid-stream"),
    )
    fake_llm = _make_fake_llm()
    fake_llm._completion = MagicMock(return_value=wrapper)

    with (
        patch("onyx.llm.multi_llm.is_true_openai_model", return_value=False),
        pytest.raises(RuntimeError, match="mid-stream"),
    ):
        LitellmLLM.invoke(fake_llm, prompt=_make_prompt())

    assert closed[0] is True


def test_stream_closes_stream_when_abandoned() -> None:
    closed = [False]
    gen = _tracking_generator(closed)
    next(gen)
    wrapper = _FakeStreamWrapper(
        _FakeProviderIterator(gen),
        chunks=[object(), object(), object()],
    )
    fake_llm = _make_fake_llm()
    fake_llm._completion = MagicMock(return_value=wrapper)
    translated_chunk = _make_stream_response("hello")

    with (
        patch("onyx.llm.multi_llm.is_true_openai_model", return_value=False),
        patch(
            "onyx.llm.model_response.from_litellm_model_response_stream",
            return_value=translated_chunk,
        ),
    ):
        stream = cast(
            Generator[ModelResponseStream, None, None],
            LitellmLLM.stream(fake_llm, prompt=_make_prompt()),
        )
        assert next(stream).choice.delta.content == "hello"
        # Caller abandons the stream mid-way (client disconnect, early return).
        stream.close()

    assert closed[0] is True


def test_stream_closes_stream_on_midstream_error() -> None:
    closed = [False]
    gen = _tracking_generator(closed)
    next(gen)
    wrapper = _FakeStreamWrapper(
        _FakeProviderIterator(gen),
        chunks=[object()],
        error=RuntimeError("provider blew up mid-stream"),
    )
    fake_llm = _make_fake_llm()
    fake_llm._completion = MagicMock(return_value=wrapper)
    translated_chunk = _make_stream_response("hello")

    with (
        patch("onyx.llm.multi_llm.is_true_openai_model", return_value=False),
        patch(
            "onyx.llm.model_response.from_litellm_model_response_stream",
            return_value=translated_chunk,
        ),
        pytest.raises(RuntimeError, match="mid-stream"),
    ):
        list(LitellmLLM.stream(fake_llm, prompt=_make_prompt()))

    assert closed[0] is True
