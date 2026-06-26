"""
agent/model_client.py
======================
The tier-resolving model client: the seam that makes local-vs-frontier a CONFIG
choice rather than a code change.

THE ONE IDEA:
The workflow calls a CAPABILITY ("classify", "synthesis"), never a model. This
client resolves capability -> ModelEntry (via config.model_for) -> endpoint, and
makes the request. Swapping a local model, or switching the frontier provider,
is an edit to config.yaml -- this file does not change, and the workflow does
not change.

WHY RAW HTTP (requests) AND NOT A VENDOR SDK:
The original demo (agent1.py) pinned itself to one vendor:
    client = anthropic.Anthropic(); MODEL = "claude-sonnet-4-5"
That is exactly the hardcoding Stage 2 removes. Pinning to ANY vendor SDK
reintroduces it. Instead we speak the ONE wire shape every provider supports --
the OpenAI-compatible /v1/chat/completions endpoint. Ollama, vLLM, Claude, and
Gemini all accept it. One request body serves all of them; the only per-provider
differences (auth header style, how tool calls are returned) are normalized
INSIDE this client and never leak to the workflow.

WHERE THIS SITS RELATIVE TO THE DEFENSES:
Below them. This client is a dumb, swappable pipe: build request, send, normalize
response. The tool registry, dispatch, HITL gate, verifier, and compaction all
live ABOVE it in the workflow layer and are model-agnostic. That ordering is the
safety property: swapping the model cannot bypass a defense, because the defenses
don't live in the swappable layer.

SECRETS:
The config carries only api_key_ref NAMES. This client resolves a name into an
actual key at call time. For now resolution is env-var based; Stage 3 replaces
the resolver with env -> keyring -> file fallback. The resolver is injected, so
that upgrade is a one-line swap and this file stays put.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from agent.config import AppConfig, ModelEntry


# ---------------------------------------------------------------------------
# Exceptions: distinct types so callers can react to different failure classes.
# ---------------------------------------------------------------------------

class ModelClientError(Exception):
    """Base class for all model-client failures."""
    pass


class SecretResolutionError(ModelClientError):
    """
    Raised when an api_key_ref cannot be resolved to an actual secret. Separated
    out because it is a SETUP problem (a missing env var), not a runtime API
    problem -- the Stage 3 preflight check will catch this before any run, but if
    it surfaces at call time we want it clearly distinguishable from a network
    error.
    """
    pass


class ModelRequestError(ModelClientError):
    """
    Raised when the HTTP request fails or the endpoint returns an error status.
    Carries the status code (when there is one) so the caller / retry logic can
    distinguish a 429 (back off and retry) from a 400 (don't retry, it's malformed).
    """
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MalformedToolCallError(ModelClientError):
    """
    Raised when the model returns a tool call we cannot parse into a clean
    (name, arguments) pair. This is the LOCAL-MODEL RELIABILITY RISK made
    concrete: an 8B model emits malformed tool calls far more often than a
    frontier model. We surface it explicitly so the workflow's dispatch can
    reject-and-retry rather than acting on garbage. Frontier models trip this
    rarely; local models are exactly why the check exists.
    """
    pass


# ---------------------------------------------------------------------------
# Secret resolver: api_key_ref NAME -> actual key. Injected, so Stage 3 swaps it.
# ---------------------------------------------------------------------------

# A resolver takes a ref name (e.g. "ANTHROPIC_API_KEY") and returns the secret,
# or None if it can't be found. Injecting this (rather than reading os.environ
# inline) is what lets Stage 3 replace env-only resolution with an
# env -> keyring -> file fallback without touching this file.
SecretResolver = Callable[[str], Optional[str]]


def env_secret_resolver(ref: str) -> Optional[str]:
    """
    Default resolver: look the ref up as an environment variable. This is the
    FIRST tier of the Stage 3 resolver (env -> keyring -> file), so shipping it
    now is not throwaway -- it's the first layer of the eventual chain.

    A local model that needs no real key (Ollama ignores it) can use any
    placeholder env var, or we tolerate a missing one for clearly-local
    endpoints (see ModelClient._resolve_key).
    """
    return os.environ.get(ref)


# ---------------------------------------------------------------------------
# Normalized response: what every provider's reply is flattened into.
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """
    One tool call the model wants to make, normalized across providers. The
    workflow feeds (name, arguments) straight into the existing tools.dispatch().
    `call_id` is echoed back when we return the tool's result, so the model can
    match result to request.
    """
    call_id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """
    A provider-agnostic response. Whatever shape Ollama/Claude/Gemini returned,
    the workflow sees THIS: some text, zero or more tool calls, and the raw
    finish reason. The workflow never inspects provider-specific JSON.

    `text` and `tool_calls` are not mutually exclusive -- a model may emit both.
    The workflow decides what to do (typically: if tool_calls, dispatch them;
    else treat text as the answer), exactly as agent1.py's _step did, but now
    against a normalized shape instead of anthropic-specific blocks.
    """
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)   # the untouched response, for logging/debug

    @property
    def wants_tools(self) -> bool:
        """True if the model asked to call at least one tool."""
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# The client.
# ---------------------------------------------------------------------------

class ModelClient:
    """
    Resolves capabilities to endpoints and makes OpenAI-compatible chat
    completion requests.

    Usage shape (workflow, a later stage):

        client = ModelClient(config)
        resp = client.complete(
            capability="classify",
            messages=[{"role": "user", "content": "..."}],
            tools=all_schemas(),          # from the existing tools.py registry
        )
        if resp.wants_tools:
            for call in resp.tool_calls:
                result = dispatch(call.name, **call.arguments)   # existing dispatch
                ...
        else:
            answer = resp.text

    The capability string is the ONLY thing tying a call to a model. Everything
    about WHICH model, WHERE it lives, and HOW to authenticate is resolved from
    config behind that string.
    """

    def __init__(
        self,
        config: AppConfig,
        secret_resolver: SecretResolver = env_secret_resolver,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config
        # Injected resolver -- Stage 3 swaps this for the layered version.
        self._resolve_secret = secret_resolver
        # A reused Session pools TCP connections across the many calls in one run
        # (100 messages -> many classify calls). Injectable so tests can pass a
        # fake. This is a performance/cleanliness detail, not a design seam.
        self._session = session or requests.Session()

    # ----- Public entry point -----

    def complete(
        self,
        capability: str,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_format: Optional[dict] = None,
    ) -> ModelResponse:
        """
        Run one chat completion for the given capability.

        `capability` resolves to a model via config (capability -> tier -> model).
        `messages` is the OpenAI-style [{"role","content"}, ...] list.
        `tools` is the schema list from the existing registry (tools.all_schemas()).
        `response_format` carries a structured-output request (e.g. JSON) for the
            spine-extraction calls; the client passes it through and the caller
            validates the parsed result.

        Returns a normalized ModelResponse. Raises ModelRequestError on HTTP
        failure, MalformedToolCallError on an unparseable tool call.

        NOTE on temperature default 0.0: the local tier's job is extract/classify/
        route -- deterministic-ish work where we want the most repeatable output,
        which also makes the verifier's life easier. Synthesis calls can override.
        """
        entry = self._config.model_for(capability)   # capability -> ModelEntry
        api_key = self._resolve_key(entry)
        payload = self._build_payload(
            entry, messages, tools, max_tokens, temperature, response_format
        )
        raw = self._post(entry, api_key, payload)
        return self._normalize(raw)

    # ----- Secret resolution -----

    def _resolve_key(self, entry: ModelEntry) -> str:
        """
        Turn the entry's api_key_ref NAME into an actual key via the injected
        resolver. We tolerate a missing key for clearly-local endpoints (Ollama
        ignores the key entirely), because forcing a user to set a dummy env var
        for their local model is needless friction. For non-local endpoints a
        missing key is a hard SecretResolutionError -- a frontier call with no
        key will only fail confusingly later, so we fail clearly now.
        """
        key = self._resolve_secret(entry.api_key_ref)
        if key:
            return key
        if self._looks_local(entry):
            # Local endpoints typically ignore the key; supply a harmless
            # placeholder so the OpenAI-compatible server gets a well-formed
            # Authorization header.
            return "local-no-key"
        raise SecretResolutionError(
            f"could not resolve secret '{entry.api_key_ref}' for model "
            f"'{entry.logical_name}'. Set it (e.g. export {entry.api_key_ref}=...) "
            "before running."
        )

    @staticmethod
    def _looks_local(entry: ModelEntry) -> bool:
        """Heuristic: is this endpoint on this machine? Used only to decide
        whether a missing key is tolerable. Same simple test the config's cost
        warning uses, kept consistent."""
        url = entry.base_url
        return "localhost" in url or "127.0.0.1" in url

    # ----- Request construction -----

    def _build_payload(
        self,
        entry: ModelEntry,
        messages: list[dict],
        tools: Optional[list[dict]],
        max_tokens: int,
        temperature: float,
        response_format: Optional[dict],
    ) -> dict:
        """
        Build the OpenAI-compatible request body. This is the ONE shape all
        providers accept. Tool schemas from the existing registry are in
        Anthropic's {name, description, input_schema} form, so we translate them
        to OpenAI's {type:function, function:{name, description, parameters}}
        form here -- the one place that translation lives, so the registry never
        has to know which provider it's feeding.
        """
        payload: dict = {
            "model": entry.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [self._tool_schema_to_openai(t) for t in tools]
            # "auto" = the model MAY call a tool but isn't forced to. This mirrors
            # agent1.py's tool_choice "auto" and preserves bounded autonomy: the
            # model isn't compelled to act every turn.
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        return payload

    @staticmethod
    def _tool_schema_to_openai(tool_schema: dict) -> dict:
        """
        Translate one registry tool schema (Anthropic-style, as tools.py emits)
        into OpenAI-style. Kept tolerant: if a schema already looks OpenAI-shaped
        (has a "function" key), pass it through untouched.
        """
        if "function" in tool_schema:
            return tool_schema
        return {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema.get("description", ""),
                # Anthropic calls it input_schema; OpenAI calls it parameters.
                "parameters": tool_schema.get("input_schema", {}),
            },
        }

    # ----- HTTP -----

    def _post(self, entry: ModelEntry, api_key: str, payload: dict) -> dict:
        """
        Send the request and return the parsed JSON body. Translates transport
        and HTTP-status failures into ModelRequestError (with status code, so
        retry logic can tell a 429 from a 400). Stage 3 adds backoff/retry around
        this; here we just classify the failure clearly.
        """
        url = entry.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._session.post(
                url, headers=headers, json=payload, timeout=entry.timeout_s
            )
        except requests.Timeout as e:
            raise ModelRequestError(
                f"request to '{entry.logical_name}' timed out after "
                f"{entry.timeout_s}s"
            ) from e
        except requests.RequestException as e:
            # Connection refused, DNS failure, etc. For a local model this most
            # often means "the Ollama/vLLM server isn't running."
            raise ModelRequestError(
                f"could not reach '{entry.logical_name}' at {url}: {e}"
            ) from e

        if resp.status_code >= 400:
            # Include a snippet of the body -- providers put the real reason there.
            body_snippet = resp.text[:500]
            raise ModelRequestError(
                f"'{entry.logical_name}' returned HTTP {resp.status_code}: "
                f"{body_snippet}",
                status_code=resp.status_code,
            )

        try:
            return resp.json()
        except ValueError as e:
            raise ModelRequestError(
                f"'{entry.logical_name}' returned non-JSON response: "
                f"{resp.text[:500]}"
            ) from e

    # ----- Response normalization -----

    def _normalize(self, raw: dict) -> ModelResponse:
        """
        Flatten an OpenAI-compatible response into our provider-agnostic
        ModelResponse. The OpenAI shape is:
            { "choices": [ { "message": { "content": ..., "tool_calls": [...] },
                             "finish_reason": ... } ] }
        We defensively handle missing pieces -- a malformed response should raise
        a clear error, never an IndexError/KeyError deep in the workflow.
        """
        choices = raw.get("choices")
        if not choices:
            raise ModelRequestError(
                f"response had no 'choices': {json.dumps(raw)[:500]}"
            )

        message = choices[0].get("message", {})
        text = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            tool_calls.append(self._parse_tool_call(tc))

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=raw,
        )

    @staticmethod
    def _parse_tool_call(tc: dict) -> ToolCall:
        """
        Parse one tool call, defending against the local-model failure mode.

        OpenAI shape:
            { "id": ..., "function": { "name": ..., "arguments": "<JSON string>" } }
        The arguments come as a JSON STRING that must be parsed. An 8B model
        frequently produces invalid JSON here (trailing commas, unquoted keys,
        truncation). We raise MalformedToolCallError so the workflow's dispatch
        can reject-and-retry rather than calling a tool with garbage arguments.
        This is the validate-reject-retry guard from the Stage 2 design, living
        at the exact boundary where bad tool calls enter the system.
        """
        function = tc.get("function") or {}
        name = function.get("name")
        if not name:
            raise MalformedToolCallError(
                f"tool call missing a function name: {json.dumps(tc)[:300]}"
            )

        raw_args = function.get("arguments", "{}")
        # Arguments may arrive as a JSON string (OpenAI/most) or already-parsed
        # dict (some local servers). Handle both.
        if isinstance(raw_args, dict):
            arguments = raw_args
        else:
            try:
                arguments = json.loads(raw_args)
            except (ValueError, TypeError) as e:
                raise MalformedToolCallError(
                    f"tool '{name}' had unparseable arguments: {raw_args!r} ({e})"
                ) from e

        if not isinstance(arguments, dict):
            raise MalformedToolCallError(
                f"tool '{name}' arguments did not parse to an object: {arguments!r}"
            )

        call_id = tc.get("id") or f"call_{name}"
        return ToolCall(call_id=call_id, name=name, arguments=arguments)
```


<invoke name="present_files">
<parameter name="filepaths">["/mnt/user-data/outputs/model_client.py"]