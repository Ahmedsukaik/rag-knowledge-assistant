"""Stand-ins for the Anthropic API in the web demo's tests. No test calls the live API."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import httpx2  # the HTTP client the anthropic SDK (1.13) is built on

USAGE = {"input_tokens": 1200, "output_tokens": 60}


def search_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in request["messages"][0]["content"] if b.get("type") == "search_result"]


def cited_reply(request: dict[str, Any], text: str | None = None, usage: dict[str, int] | None = None) -> dict[str, Any]:
    """A reply that states the first sentence of the first search result, citing it, as the API returns it."""
    first = search_results(request)[0]
    sentence = first["content"][0]["text"]
    return {"model": request["model"], "stop_reason": "end_turn", "usage": dict(usage or USAGE), "content": [
        {"type": "text", "text": text or sentence, "citations": [{
            "type": "search_result_location", "cited_text": sentence, "search_result_index": 0, "source": first["source"],
            "title": first["title"], "start_block_index": 0, "end_block_index": 1}]}]}


class FakeAnthropic:
    """client.beta.messages.create(**request) returns `reply(request)`, or raises it if it is an exception."""

    def __init__(self, reply: Callable[[dict[str, Any]], Any] | BaseException = cited_reply):
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply(request)


def api_error(kind: str) -> BaseException:
    """The exception the anthropic SDK raises for each kind of failure."""
    import anthropic

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def response(status: int) -> httpx2.Response:
        return httpx2.Response(status, request=request)

    if kind == "budget":
        body = {"type": "error", "error": {"type": "rate_limit_error", "message": "enforced_spend_limit_reached"}}
        return anthropic.RateLimitError("spend limit", response=response(429), body=body)
    if kind == "budget_usage":
        body = {"type": "error", "error": {"type": "invalid_request_error",
                                           "message": "You have reached your specified API usage limits."}}
        return anthropic.BadRequestError("usage limits", response=response(400), body=body)
    if kind == "rate_limited":
        return anthropic.RateLimitError("rate limited", response=response(429),
                                        body={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    if kind == "auth":
        return anthropic.AuthenticationError("bad key", response=response(401),
                                             body={"type": "error", "error": {"type": "authentication_error"}})
    if kind == "timeout":
        return anthropic.APITimeoutError(request=request)
    if kind == "error":
        return anthropic.InternalServerError("overloaded", response=response(529),
                                             body={"type": "error", "error": {"type": "overloaded_error"}})
    raise ValueError(kind)
