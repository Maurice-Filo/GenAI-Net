import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class SymbolRef:
    """Resolved reference to either an input or a species index.

    Attributes:
        name: Symbol name (e.g., "u_1", "X_1").
        domain: Either "input" or "species".
        index: Zero-based index into u (if input) or x0 (if species).
    """
    name: str
    domain: str  # "input" | "species"
    index: int


def resolve_named_symbols(
    *,
    template_crn: Any,
    names: Iterable[str],
    input_dict_attr: str = "input_idx_dict",
    species_dict_attr: str = "species_idx_dict",
) -> Dict[str, SymbolRef]:
    """Resolve symbol names against IOCRN input/species dictionaries.

    Resolution order:
      1) template_crn.input_idx_dict
      2) template_crn.species_idx_dict

    Args:
        template_crn: IOCRN object with `.input_idx_dict` and `.species_idx_dict`.
        names: Iterable of symbol names to resolve.
        input_dict_attr: Attribute name for input map dict.
        species_dict_attr: Attribute name for species map dict.

    Returns:
        Dict mapping name -> SymbolRef.

    Raises:
        AttributeError: If dictionaries are missing on template_crn.
        KeyError: If any name cannot be resolved.
    """
    input_idx_dict = getattr(template_crn, input_dict_attr)
    species_idx_dict = getattr(template_crn, species_dict_attr)

    unresolved: List[str] = []
    out: Dict[str, SymbolRef] = {}

    for name in names:
        if name in input_idx_dict:
            out[name] = SymbolRef(name=name, domain="input", index=int(input_idx_dict[name]))
            continue
        if name in species_idx_dict:
            out[name] = SymbolRef(name=name, domain="species", index=int(species_idx_dict[name]))
            continue
        unresolved.append(name)

    if unresolved:
        available_inputs = sorted(list(input_idx_dict.keys()))
        available_species = sorted(list(species_idx_dict.keys()))
        raise KeyError(
            "Unresolved symbols in target expression: "
            f"{unresolved}. "
            "Expected names to match either template_crn.input_idx_dict "
            "or template_crn.species_idx_dict. "
            f"Available inputs: {available_inputs}. "
            f"Available species: {available_species}."
        )

    return out


def _callable_param_names(fn: Callable[..., Any]) -> List[str]:
    """Extract explicit parameter names from a callable.

    Notes:
        - Supports plain functions and lambdas.
        - If **kwargs is present, we cannot infer names reliably, so we disallow it.

    Args:
        fn: Callable.

    Returns:
        List of parameter names in order.

    Raises:
        ValueError: If callable uses *args/**kwargs or has zero parameters.
    """
    sig = inspect.signature(fn)
    names: List[str] = []

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                "Target callable must have explicit named parameters (no *args/**kwargs), "
                f"got signature: {sig}"
            )
        names.append(p.name)

    if not names:
        raise ValueError(f"Target callable must accept at least one argument, got signature: {sig}")

    return names


def build_named_target_callable(
    *,
    fn: Callable[..., Any],
    template_crn: Any,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build a target function that evaluates a user expression using semantic names.

    The user provides:
        target = lambda u_1, X_1, u_2: (u_1 + u_2) * X_1

    This builder:
      - inspects parameter names: ["u_1", "X_1", "u_2"]
      - resolves each name to input/species index using template_crn dicts
      - returns a callable g(u, x0) -> float

    Args:
        fn: User callable with explicit named parameters.
        template_crn: IOCRN with `.input_idx_dict` and `.species_idx_dict`.

    Returns:
        Callable mapping (u, x0) -> float.

    Raises:
        ValueError: If signature invalid.
        KeyError: If names cannot be resolved.
    """
    param_names = _callable_param_names(fn)
    refs = resolve_named_symbols(template_crn=template_crn, names=param_names)

    def g(u: np.ndarray, x0: np.ndarray) -> float:
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)

        kwargs: Dict[str, float] = {}
        for name in param_names:
            ref = refs[name]
            if ref.domain == "input":
                kwargs[name] = float(u[ref.index])
            else:
                kwargs[name] = float(x0[ref.index])

        val = fn(**kwargs)
        return float(val)

    return g


from typing import Any, Callable, List, Optional, Sequence, Union
import numpy as np

TargetSpec = Union[float, int, str, Callable[..., Any]]


def build_r_list_from_target(
    *,
    target: TargetSpec,
    template_crn: Any,
    u_list: Sequence[np.ndarray],
    x0_list: Optional[Sequence[np.ndarray]] = None,
    expand_over_ic: bool = False,
    q: int = 1,
) -> List[np.ndarray]:
    """Build r_list for tracking-style objectives from a target specification.

    Supports:
      - constant float/int target
      - callable target with named arguments resolved via
        template_crn.input_idx_dict / template_crn.species_idx_dict

    Args:
        target: float/int or callable (named args).
        template_crn: IOCRN used for name->index resolution.
        u_list: List of inputs, each shape (p,).
        x0_list: Optional list of IC vectors (needed if callable references species).
        expand_over_ic: If True, returns r_list aligned with (x0,u) nested loops:
            for x0 in x0_list:
              for u in u_list:
                append r(u,x0)
          If False, returns per-u list aligned with u_list only.
        q: Output dimension (usually 1). Currently enforced as q==1.

    Returns:
        List of np.ndarray targets, each shape (q,).

    Raises:
        ValueError: for unknown target type or missing x0_list when needed.
    """
    if q != 1:
        raise ValueError("build_r_list_from_target currently supports q==1 only.")

    u_list = [np.asarray(u, dtype=np.float32).reshape(-1) for u in u_list]

    # --- callable / named-expression target ---
    if callable(target):
        # g(u, x0) -> float
        g = build_named_target_callable(fn=target, template_crn=template_crn)

        if expand_over_ic:
            if x0_list is None:
                raise ValueError("expand_over_ic=True requires x0_list.")
            r_list: List[np.ndarray] = []
            for x0 in x0_list:
                x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
                for u in u_list:
                    r_list.append(np.array([g(u, x0)], dtype=np.float32))
            return r_list

        # per-u only: choose a representative x0 (first) if given, else zeros
        if x0_list is None or len(x0_list) == 0:
            # if user referenced species, build_named_target_callable will index x0;
            # so provide a safe x0 of correct length if possible.
            # If template_crn has species_idx_dict, we can infer size from max index.
            species_idx_dict = getattr(template_crn, "species_idx_dict", {})
            if species_idx_dict:
                n_species = max(int(i) for i in species_idx_dict.values()) + 1
                x0_ref = np.zeros((n_species,), dtype=np.float32)
            else:
                x0_ref = np.zeros((1,), dtype=np.float32)
        else:
            x0_ref = np.asarray(x0_list[0], dtype=np.float32).reshape(-1)

        return [np.array([g(u, x0_ref)], dtype=np.float32) for u in u_list]

    # --- constant target ---
    if isinstance(target, (int, float)):
        c = float(target)
        return [np.array([c], dtype=np.float32) for _ in u_list]

    raise ValueError(
        "target must be a float/int or a callable with named args "
        "(resolved via template_crn.input_idx_dict/species_idx_dict)."
    )
