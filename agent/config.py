"""
agent/config.py
===============
Loads config.yaml into validated, typed Python objects.

WHAT THIS FILE IS FOR:
The YAML file is the single human-facing surface. This module is the adapter
between that file and the typed objects the rest of the code already speaks --
notably the VerifierConfig and CompactionConfig the defense modules accept. We
do NOT invent a parallel config representation; we read the YAML and hand back
the exact types Stage 1 already defined, plus a few new ones for the model
registry and tier map that Stage 2 introduces.

WHY A VALIDATING LOADER (not just yaml.safe_load and go):
Two reasons.
  1. YAML has footguns -- the famous "Norway problem" (`no` -> False), silent
     type coercion, missing keys surfacing as None deep inside the code. We
     validate up front so a malformed config fails LOUDLY at load with a clear
     message, instead of quietly mis-running at 6am.
  2. The tier table doubles as the escalation boundary (Stage 2). A careless
     edit -- pointing `classify` at a frontier key -- silently changes your cost
     posture. The loader warns on exactly that, the cheap guard we flagged in
     the Stage 2 design.

This file does NOT resolve secrets. It reads api_key_ref NAMES only. Actual
secret resolution is Stage 3 -- the config references a secret by name and never
contains the value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# We reuse the EXACT config types the defense modules already accept, so the
# defenses need zero changes. The loader's job is to populate these from YAML.
from defenses.verifier import VerifierConfig
from defenses.compaction import CompactionConfig


# ---------------------------------------------------------------------------
# Exceptions: a clear, dedicated error for bad config.
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """
    Raised when the config is missing, malformed, or internally inconsistent.
    We use one dedicated exception type so the entrypoint (and the Stage 3
    preflight check) can catch config problems specifically and print a helpful
    "fix your config.yaml" message rather than a raw KeyError/TypeError traceback.
    """
    pass


# ---------------------------------------------------------------------------
# Block A: the model registry.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """
    One entry in the model registry -- pure connection info for one endpoint.
    Frozen because, once loaded, a model's connection details are fixed for the
    run; nothing should mutate an endpoint mid-run.

    api_key_ref is the NAME of a secret (an env var name, typically), never the
    secret itself. The model client (next file) resolves it at call time.
    """
    logical_name: str        # the key used everywhere else, e.g. "local-8b"
    base_url: str
    model_name: str
    context_window: int
    api_key_ref: str
    timeout_s: int

    def __post_init__(self) -> None:
        # Validate the numeric fields are sane. A context_window of 0 or a
        # negative timeout is a config typo we want caught at load, not when the
        # first API call constructs a nonsense request.
        if self.context_window <= 0:
            raise ConfigError(
                f"model '{self.logical_name}': context_window must be positive, "
                f"got {self.context_window}"
            )
        if self.timeout_s <= 0:
            raise ConfigError(
                f"model '{self.logical_name}': timeout_s must be positive, "
                f"got {self.timeout_s}"
            )
        for field_name in ("base_url", "model_name", "api_key_ref"):
            if not getattr(self, field_name):
                raise ConfigError(
                    f"model '{self.logical_name}': {field_name} is required and "
                    "cannot be empty"
                )


# ---------------------------------------------------------------------------
# Block B: the tier map (capability -> logical model name).
# ---------------------------------------------------------------------------

# The capabilities the workflow knows how to request. Declaring them as a
# constant (rather than accepting any string) means a typo in the YAML tier
# block -- "classifyy: local-8b" -- is caught at load instead of failing later
# when the workflow asks for the "classify" tier and finds nothing.
KNOWN_CAPABILITIES: frozenset[str] = frozenset({
    "classify",
    "prioritize",
    "summarize",
    "route",
    "thread_deep_read",
    "synthesis",
})

# The capabilities that are cheap and high-frequency. Pointing any of these at a
# frontier model is legal but almost always a mistake (it runs up cost). The
# loader warns -- it does not block, because you might genuinely want it.
HIGH_FREQUENCY_CAPABILITIES: frozenset[str] = frozenset({
    "classify",
    "prioritize",
    "summarize",
    "route",
})


@dataclass(frozen=True)
class TierMap:
    """
    The capability -> logical-model-name mapping. This IS the static escalation
    boundary: which jobs run local vs. frontier is entirely determined here.

    Stored as a plain dict inside a frozen wrapper. `resolve` is the single
    lookup the model client uses to turn a capability into a model key.
    """
    mapping: dict[str, str]

    def resolve(self, capability: str) -> str:
        """Return the logical model name bound to a capability, or raise."""
        if capability not in self.mapping:
            raise ConfigError(
                f"no model configured for capability '{capability}'. "
                f"Configured: {sorted(self.mapping)}"
            )
        return self.mapping[capability]


# ---------------------------------------------------------------------------
# Block D: workflow settings.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkflowConfig:
    """
    The user-facing run settings. vip_senders is normalized to lowercase at load
    so the VIP match (in extraction and the verifier) is reliably
    case-insensitive without every call site remembering to lowercase.
    """
    vip_senders: frozenset[str]
    morning_run_time: str
    debrief_run_time: str
    read_scope: str
    failure_mode: str

    def is_vip(self, sender: Optional[str]) -> bool:
        """True if a sender address is on the VIP list (case-insensitive)."""
        if not sender:
            return False
        return sender.strip().lower() in self.vip_senders


# ---------------------------------------------------------------------------
# The top-level config object: everything, assembled and validated.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """
    The whole validated config. This is what the entrypoint loads once and hands
    to the client, the defenses, and the workflow.

    Note the defense configs (verifier, compaction) and the autonomy caps are
    surfaced here as the SAME types Stage 1 defined, so wiring them into the
    defenses later is a direct pass-through with no translation.
    """
    models: dict[str, ModelEntry]      # logical_name -> entry
    tiers: TierMap
    workflow: WorkflowConfig

    # Defense configs, ready to hand straight to the Stage 1 modules.
    verifier: VerifierConfig
    compaction: CompactionConfig
    investigation_fetch_cap: int
    write_action_cap: int
    verifier_max_retries: int

    # Warnings collected during load (non-fatal). The entrypoint/preflight prints
    # these. Kept on the config so a caller can inspect them programmatically too.
    warnings: tuple[str, ...] = ()

    def model_for(self, capability: str) -> ModelEntry:
        """
        Convenience: capability -> the actual ModelEntry to call. Combines the
        tier lookup (capability -> logical name) with the registry lookup
        (logical name -> entry). This is the single call the model client makes.
        """
        logical = self.tiers.resolve(capability)
        if logical not in self.models:
            raise ConfigError(
                f"capability '{capability}' is mapped to model '{logical}', "
                f"which is not defined in the models block. "
                f"Defined: {sorted(self.models)}"
            )
        return self.models[logical]


# ---------------------------------------------------------------------------
# The loader: YAML file -> validated AppConfig.
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> AppConfig:
    """
    Read, parse, and validate a config file. Raises ConfigError with a specific
    message on any problem. Returns a fully-validated AppConfig on success.

    The validation order is deliberate: structure first (is each block present
    and the right type?), then contents (are the values sane?), then
    cross-references (does every tier point at a real model?). Later checks
    assume earlier ones passed, so we fail at the most specific cause.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    # safe_load (not load) -- never execute arbitrary tags from a config file.
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"config file is not valid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping of the four blocks")

    warnings: list[str] = []

    # --- Block A: models ---
    models = _load_models(raw)

    # --- Block B: tiers (validated against the models we just loaded) ---
    tiers = _load_tiers(raw, models, warnings)

    # --- Block C: thresholds -> defense config objects ---
    verifier_cfg, compaction_cfg, inv_cap, write_cap, max_retries = _load_thresholds(raw)

    # --- Block D: workflow ---
    workflow = _load_workflow(raw)

    return AppConfig(
        models=models,
        tiers=tiers,
        workflow=workflow,
        verifier=verifier_cfg,
        compaction=compaction_cfg,
        investigation_fetch_cap=inv_cap,
        write_action_cap=write_cap,
        verifier_max_retries=max_retries,
        warnings=tuple(warnings),
    )


# ----- Per-block loaders (private helpers) -----

def _require_block(raw: dict, name: str) -> dict:
    """Fetch a top-level block, raising a clear error if missing or wrong type."""
    block = raw.get(name)
    if block is None:
        raise ConfigError(f"missing required config block: '{name}'")
    if not isinstance(block, dict):
        raise ConfigError(f"config block '{name}' must be a mapping")
    return block


def _load_models(raw: dict) -> dict[str, ModelEntry]:
    """Build the model registry from Block A, validating each entry."""
    block = _require_block(raw, "models")
    if not block:
        raise ConfigError("'models' block is empty; define at least one model")

    models: dict[str, ModelEntry] = {}
    for logical_name, entry in block.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"model '{logical_name}' must be a mapping of fields")
        try:
            # Construction validates via ModelEntry.__post_init__. We pull each
            # field explicitly so a missing key is a clear ConfigError, not a
            # confusing TypeError about missing arguments.
            models[logical_name] = ModelEntry(
                logical_name=logical_name,
                base_url=_req(entry, "base_url", logical_name),
                model_name=_req(entry, "model_name", logical_name),
                context_window=_req(entry, "context_window", logical_name),
                api_key_ref=_req(entry, "api_key_ref", logical_name),
                timeout_s=_req(entry, "timeout_s", logical_name),
            )
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"model '{logical_name}': {e}") from e
    return models


def _load_tiers(
    raw: dict, models: dict[str, ModelEntry], warnings: list[str]
) -> TierMap:
    """
    Build and validate the tier map from Block B. Two validations beyond
    structure:
      1. Every tier value names a model that exists in Block A (cross-reference).
      2. Every key is a known capability, and every known capability is mapped
         (no typos, no gaps).
    Plus the cost-posture WARNING: a high-frequency capability pointed at a
    frontier model.
    """
    block = _require_block(raw, "tiers")
    mapping: dict[str, str] = {}

    for capability, logical_name in block.items():
        # Typo guard: reject unknown capability keys.
        if capability not in KNOWN_CAPABILITIES:
            raise ConfigError(
                f"unknown capability '{capability}' in tiers block. "
                f"Known: {sorted(KNOWN_CAPABILITIES)}"
            )
        # Cross-reference: the named model must exist.
        if logical_name not in models:
            raise ConfigError(
                f"capability '{capability}' points at model '{logical_name}', "
                f"which is not defined in the models block. "
                f"Defined: {sorted(models)}"
            )
        mapping[capability] = logical_name

    # Gap guard: every known capability must be mapped, or the workflow will ask
    # for a tier that doesn't exist mid-run.
    missing = KNOWN_CAPABILITIES - set(mapping)
    if missing:
        raise ConfigError(
            f"these capabilities are not mapped in the tiers block: "
            f"{sorted(missing)}"
        )

    # Cost-posture warning (non-fatal): high-frequency work on a frontier model.
    # We detect "frontier" heuristically by the model's logical name containing
    # "frontier" OR its base_url not being localhost. Kept simple and explicit.
    for capability in HIGH_FREQUENCY_CAPABILITIES:
        logical_name = mapping[capability]
        entry = models[logical_name]
        looks_frontier = (
            "frontier" in logical_name.lower()
            or "localhost" not in entry.base_url
            and "127.0.0.1" not in entry.base_url
        )
        if looks_frontier:
            warnings.append(
                f"capability '{capability}' is high-frequency but is pointed at "
                f"'{logical_name}', which looks like a non-local model. This will "
                f"run up cost. Intentional? If not, point it at a local model."
            )

    return TierMap(mapping=mapping)


def _load_thresholds(
    raw: dict,
) -> tuple[VerifierConfig, CompactionConfig, int, int, int]:
    """
    Read Block C and produce the defense config objects directly. This is the
    adapter's core job: the YAML's flat thresholds become the structured config
    types the Stage 1 modules already accept.
    """
    block = _require_block(raw, "thresholds")

    # Bounded autonomy caps.
    inv_cap = _req_int(block, "investigation_fetch_cap")
    write_cap = _req_int(block, "write_action_cap")

    # Compaction -> CompactionConfig (the exact type compaction.py expects).
    compaction_cfg = CompactionConfig(
        trigger_chars=_req_int(block, "compaction_trigger_chars"),
        target_chars=_req_int(block, "compaction_target_chars"),
        skip_vip=_req_bool(block, "compaction_skip_vip"),
    )

    # Verifier -> VerifierConfig. urgency_phrases is a list in YAML; the config
    # type wants a tuple (it's frozen), so we convert. We also lowercase the
    # phrases here so the floor's matching is reliably case-insensitive.
    phrases_raw = block.get("verifier_urgency_phrases")
    if not isinstance(phrases_raw, list) or not phrases_raw:
        raise ConfigError(
            "thresholds.verifier_urgency_phrases must be a non-empty list"
        )
    phrases = tuple(str(p).lower() for p in phrases_raw)

    verifier_cfg = VerifierConfig(
        temporal_window_days=_req_int(block, "verifier_temporal_window_days"),
        urgency_phrases=phrases,
        urgency_window_hours=_req_int(block, "verifier_urgency_window_hours"),
    )

    max_retries = _req_int(block, "verifier_max_retries")

    return verifier_cfg, compaction_cfg, inv_cap, write_cap, max_retries


def _load_workflow(raw: dict) -> WorkflowConfig:
    """Read and validate Block D."""
    block = _require_block(raw, "workflow")

    vips_raw = block.get("vip_senders") or []
    if not isinstance(vips_raw, list):
        raise ConfigError("workflow.vip_senders must be a list of email addresses")
    # Normalize to lowercase once, here, so every VIP comparison downstream is
    # case-insensitive for free.
    vips = frozenset(str(v).strip().lower() for v in vips_raw)

    read_scope = block.get("read_scope", "headers_snippets")
    if read_scope != "headers_snippets":
        raise ConfigError(
            f"workflow.read_scope only supports 'headers_snippets' in v1, "
            f"got '{read_scope}'"
        )

    failure_mode = block.get("failure_mode", "partial_flagged")
    if failure_mode != "partial_flagged":
        raise ConfigError(
            f"workflow.failure_mode only supports 'partial_flagged' in v1, "
            f"got '{failure_mode}'"
        )

    return WorkflowConfig(
        vip_senders=vips,
        morning_run_time=_req_str(block, "morning_run_time", "workflow"),
        debrief_run_time=_req_str(block, "debrief_run_time", "workflow"),
        read_scope=read_scope,
        failure_mode=failure_mode,
    )


# ----- Tiny typed-getter helpers: turn missing/wrong-type keys into clear errors -----
# These exist so every "missing key" or "wrong type" failure reads as a specific
# ConfigError naming the field, instead of a KeyError or TypeError from deep in
# construction. They are the difference between "fix thresholds.write_action_cap"
# and an opaque traceback at 6am.

def _req(block: dict, key: str, owner: str):
    """Require a key to be present (any type). Used where __post_init__ will
    further validate the value."""
    if key not in block:
        raise ConfigError(f"'{owner}': missing required field '{key}'")
    return block[key]


def _req_int(block: dict, key: str) -> int:
    val = block.get(key)
    # bool is a subclass of int in Python; reject it explicitly so `true` in a
    # numeric field doesn't silently become 1.
    if not isinstance(val, int) or isinstance(val, bool):
        raise ConfigError(f"'{key}' must be an integer, got {val!r}")
    return val


def _req_bool(block: dict, key: str) -> bool:
    val = block.get(key)
    if not isinstance(val, bool):
        raise ConfigError(f"'{key}' must be a boolean (true/false), got {val!r}")
    return val


def _req_str(block: dict, key: str, owner: str) -> str:
    val = block.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ConfigError(f"'{owner}': '{key}' must be a non-empty string, got {val!r}")
    return val