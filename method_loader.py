from __future__ import annotations

import importlib
import inspect
from typing import Any, Iterable

from base_method import BaseMethod

# Mapping of normalized method names to their default python paths.
# Need manual update
DEFAULT_METHOD_MODULES = {
    "selfcrit": "methods.selfcrit_method",
    "minkprob": "methods.minkprob_method",
    "loss_zlib_lowercase": "methods.loss_zlib_lowercase_method",
    "dcpdd": "methods.dcpdd_method",
    "recall": "methods.recall_method",
    "camia": "methods.camia_method",
    "pac": "methods.pac_method",
    "guided_instruction": "methods.guided_instruction_method",
    "dcq": "methods.dcq_method",
    "neighborhood_attack": "methods.neighborhood_attack_method",
    "cdd": "methods.cdd_method"
}

# Aliases for method names.
# Need manual update
METHOD_ALIASES = {
    "selfcrit": "selfcrit",
    "minkprob": "minkprob",
    "minkpp": "minkprob",
    "perplexity": "loss_zlib_lowercase",
    "zlib": "loss_zlib_lowercase",
    "lowercase": "loss_zlib_lowercase",
    "dcpdd": "dcpdd",
    "recall": "recall",
    "camia": "camia",
    "pac": "pac",
    "guided_instruction": "guided_instruction", 
    "dcq": "dcq", 
    "neighborhood_attack": "neighborhood_attack", 
    "cdd": "cdd" #
}


def normalize_method_name(name: str) -> str:
    normalized = name.strip().lower()
    return METHOD_ALIASES.get(normalized, normalized)


def _module_name_for(method_name: str) -> str:
    if method_name in DEFAULT_METHOD_MODULES:
        return DEFAULT_METHOD_MODULES[method_name]
    return f"methods.{method_name}_method"


def _filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return {}

    signature = inspect.signature(callable_obj)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in params}


def _method_kwargs_for(
    method_name: str,
    method_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not method_kwargs:
        return {}
    key = normalize_method_name(method_name)
    # Optional wildcard kwargs applied to all methods.
    wildcard_kwargs = method_kwargs.get("*", {})
    specific_kwargs = method_kwargs.get(key, {})
    if not wildcard_kwargs:
        return specific_kwargs
    if not specific_kwargs:
        return wildcard_kwargs
    merged = dict(wildcard_kwargs)
    merged.update(specific_kwargs)
    return merged


def load_method(
    method_name: str,
    method_kwargs: dict[str, dict[str, Any]] | None = None,
) -> BaseMethod:
    key = normalize_method_name(method_name)
    module_name = _module_name_for(key)
    module = importlib.import_module(module_name)
    kwargs_for_method = _method_kwargs_for(key, method_kwargs)

    factory = getattr(module, "build_method", None)
    if callable(factory):
        method = factory(**_filtered_kwargs(factory, kwargs_for_method))
        if isinstance(method, BaseMethod):
            return method
        raise TypeError(f"{module_name}.build_method() did not return BaseMethod")

    for value in module.__dict__.values():
        if isinstance(value, type) and issubclass(value, BaseMethod) and value is not BaseMethod:
            return value(**_filtered_kwargs(value, kwargs_for_method))

    raise ValueError(
        f"No method implementation found in {module_name}. "
        f"Define build_method() or a BaseMethod subclass."
    )


def load_methods(
    method_names: Iterable[str],
    method_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[BaseMethod]:
    methods: list[BaseMethod] = []
    seen_normalized: set[str] = set()
    for method_name in method_names:
        normalized = normalize_method_name(method_name)
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        methods.append(load_method(normalized, method_kwargs=method_kwargs))
    return methods
