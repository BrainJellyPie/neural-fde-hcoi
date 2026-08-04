"""Shared plotting configuration."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#0d9488"]

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def save(fig, path_without_extension: str) -> None:
    fig.savefig(f"{path_without_extension}.pdf")
    fig.savefig(f"{path_without_extension}.png")
    plt.close(fig)
