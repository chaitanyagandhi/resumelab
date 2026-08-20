"""Token accounting taken from the wire rather than from parsed responses.

Both provider SDKs validate a structured response into its Pydantic model *inside*
the call that fetches it. When that validation fails the exception escapes the SDK
call, so any accounting written after it never runs, and the attempt disappears from
the totals. It is still billed: the model read the prompt and wrote an answer, and
being unable to use the answer does not refund it.

Measured against a provider console on one run: 73,778 tokens recorded against
112,393 actually charged, a third of the spend invisible. Every missing token was a
schema repair, which is exactly the case this project retries.

So usage is read from the HTTP response as it arrives, before anything tries to
interpret the body. That counts what the provider counted, which is the number the
invoice is based on, and it does not care what either SDK does with the payload
afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx

from resumelab.llm.client import TokenUsage

logger = logging.getLogger(__name__)

_USAGE_FIELDS = (
    # OpenAI, then Anthropic. Reading both means neither adapter needs its own hook.
    ("prompt_tokens", "completion_tokens"),
    ("input_tokens", "output_tokens"),
)


def usage_of(payload: object) -> TokenUsage | None:
    """Read token usage out of a decoded response body, whichever provider sent it.

    A body that carries no usage block still returns a zero count rather than
    nothing: the call happened and is worth counting, the provider simply said
    nothing about its size. Only a body that is not a JSON object at all returns
    ``None``, because there is no evidence a provider call is what produced it.
    """
    if not isinstance(payload, dict):
        return None
    reported = payload.get("usage")
    if not isinstance(reported, dict):
        return TokenUsage()

    for prompt_key, completion_key in _USAGE_FIELDS:
        if prompt_key in reported or completion_key in reported:
            prompt = _count(reported.get(prompt_key))
            completion = _count(reported.get(completion_key))
            total = _count(reported.get("total_tokens")) or prompt + completion
            return TokenUsage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
            )
    # A usage block in a shape neither provider uses. The call still happened, so it
    # is counted; only its size is unknown.
    return TokenUsage()


def _count(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def metering_http_client(
    record: Callable[[TokenUsage], None],
    *,
    timeout: float,
) -> httpx.Client:
    """An HTTP client that reports the usage on every response it carries.

    The hook reads the body itself. That is safe for the requests this project makes,
    which are all unstreamed JSON, and httpx caches the content so the SDK reading it
    afterwards costs nothing. Anything that cannot be decoded is passed over in
    silence: a hook that raised would turn a provider hiccup into a crash, and it has
    no business failing a request it only exists to observe.
    """

    def on_response(response: httpx.Response) -> None:
        # Only what the provider actually served. A rejected request bills nothing,
        # and counting it would overstate a run in the opposite direction.
        if not response.is_success:
            return
        try:
            response.read()
            usage = usage_of(response.json())
        except Exception:
            logger.debug("could not read usage from a response", exc_info=True)
            return
        if usage is not None:
            record(usage)

    return httpx.Client(event_hooks={"response": [on_response]}, timeout=timeout)
