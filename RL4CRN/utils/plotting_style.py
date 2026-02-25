import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Any, Dict, Optional, Tuple, List


# -------------------------
# Paper-ready plotting tools
# -------------------------

_DEFAULT_PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,          # ↓ from 7
    "axes.titleweight": "regular",# ↓ from bold (optional but helps)
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,   # ↑ from 0.02
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_DEFAULT_PLOT_CFG = {
    "rc": {},
    "figsize": (3.2*2, 2.2*2),        # sensible single-panel default
    "constrained_layout": True,   # ↑
    "tight_layout": False,        # ↓
    "grid": True,
    "grid_kwargs": {"alpha": 0.25, "linewidth": 0.6},
    "despine": True,
    "spine_lw": 0.8,
    "alpha": None,
    "lw": None,
    "title": None,
    "xlabel": None,
    "ylabel": None,
    "title_pad": 5.0,             # ↑
    "legend": "auto",
    "legend_kwargs": {"frameon": True, "handlelength": 2.0},

    # NEW: suptitle controls (fixes “title too large”)
    "suptitle": None,
    "suptitle_size": 7,
    "suptitle_weight": "bold",
    "suptitle_y": 0.99,

    "save": None,
}

def _merge_cfg(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow merge for plot config dictionaries (one-level nested dict merge)."""
    if override is None:
        return dict(base)
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


class paper_rc_context:
    """Context manager applying paper-ready rcParams plus user overrides."""

    def __init__(self, rc_override: Optional[Dict[str, Any]] = None):
        self.rc = dict(_DEFAULT_PAPER_RC)
        if rc_override:
            self.rc.update(rc_override)
        self._ctx = None

    def __enter__(self):
        self._ctx = mpl.rc_context(self.rc)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def _ensure_axes_list(axes, n: int):
    """Ensure axes is a flat list of length n."""
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    axes = list(np.ravel(axes))
    if len(axes) != n:
        raise ValueError(f"Expected {n} axes, got {len(axes)}")
    return axes


def _apply_axes_style(ax, cfg: Dict[str, Any]):
    """Apply consistent grid/spines cosmetics."""
    if cfg.get("grid", True):
        ax.grid(True, which="major", **cfg.get("grid_kwargs", {}))
    if cfg.get("despine", True):
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if hasattr(ax, "spines"):
        for s in ax.spines.values():
            s.set_linewidth(cfg.get("spine_lw", 0.8))


def _maybe_legend(ax, n_series: int, cfg: Dict[str, Any]):
    """Show legend depending on cfg and number of plotted series."""
    leg = cfg.get("legend", "auto")
    if leg is False:
        return
    if leg == "auto" and n_series > 10:
        return
    if leg is True or leg == "auto":
        ax.legend(**cfg.get("legend_kwargs", {}))


def _maybe_save(fig, cfg: Dict[str, Any]):
    """Save figure if cfg['save'] contains a path."""
    save = cfg.get("save")
    if not save:
        return
    path = save.get("path")
    if not path:
        return
    extra = dict(save)
    extra.pop("path", None)
    fig.savefig(path, **extra)