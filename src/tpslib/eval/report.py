"""Shared plotting style for phase reports (static matplotlib figures)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, fixed order (validated slots 1-5); recessive ink and grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"


def style(ax, title: str | None = None):
    """Apply the report style (recessive grid and axes) and an optional left-aligned title."""
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=10)


def save(fig, path):
    """Save a figure at 150 dpi with tight layout and close it."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
