"""Strict adapter for the existing capitalized FACED full-source prior schema.

The original qualification loader is frozen and remains unchanged. This
adapter accepts the producer's exact ``Negative``/``Positive`` coarse keys;
it neither normalizes alternative schemas nor changes any saved value. All
numeric model-state loading and inference still use the frozen backend.
"""
from __future__ import annotations

from collections.abc import Mapping

from .faced_frozen_inference import (
    CATEGORIES, DEVELOPMENT_CLIPS, METHODS, FrozenBundle,
    _bound_bytes, _hash, _json, _number, load_frozen_state,
)


def _qualify_prior(prior):
    """Require the precise existing source schema and finite bounded medians."""
    expected = {
        "coarse": {"Negative", "Positive"},
        "category": set(CATEGORIES),
        "video": {str(v) for v in DEVELOPMENT_CLIPS},
    }
    if not isinstance(prior, dict) or set(prior) != {"global", *expected}:
        raise ValueError("Unexpected capitalized full-source prior schema")
    # The complete schema/support barrier precedes numeric prior validation.
    for field, keys in expected.items():
        if not isinstance(prior[field], dict) or set(prior[field]) != keys:
            raise ValueError("Unexpected capitalized full-source prior support")
    _number(prior["global"], "global prior", 0, 7)
    for field in expected:
        for value in prior[field].values():
            _number(value, field + " prior", 0, 7)
    return prior


def load_frozen_bundle_v2(state_bindings, prior_path, prior_sha256,
                          source_identity_sha256):
    """Load the same three source states and original uppercase prior bytes.

    Each state binding retains the frozen backend's ``path``, ``sha256`` and
    original ``record`` requirements. No fitting, label loading, recording
    access, schema conversion or prior rewriting is performed here.
    """
    _hash(prior_sha256)
    _hash(source_identity_sha256)
    if (not isinstance(state_bindings, Mapping)
            or set(state_bindings) != set(METHODS[1:])):
        raise PermissionError("Exactly three fitted state bindings are required")
    # Qualify every binding before opening the first state file.
    for model in METHODS[1:]:
        binding = state_bindings[model]
        if (not isinstance(binding, Mapping)
                or set(binding) != {"path", "sha256", "record"}
                or not isinstance(binding["record"], Mapping)
                or binding["record"].get("model") != model):
            raise ValueError("Invalid model state binding")
    states = {
        model: load_frozen_state(
            state_bindings[model]["path"],
            state_bindings[model]["sha256"],
            state_bindings[model]["record"],
            source_identity_sha256,
        )
        for model in METHODS[1:]
    }
    prior = _qualify_prior(_json(_bound_bytes(prior_path, prior_sha256, 65536)))
    return FrozenBundle(states, prior["category"], prior_sha256,
                        source_identity_sha256)
