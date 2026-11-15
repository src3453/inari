"""Yellow-based colour themes (cycle with `t` in the monitor)."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.theme import Theme as RichTheme


@dataclass(frozen=True)
class Palette:
    name: str
    accent: str  # main yellow: titles, active tab, bars
    accent2: str  # lighter yellow: secondary highlights
    text: str  # body text
    dim: str  # labels, borders
    track: str  # empty part of bars
    good: str
    warn: str
    bad: str
    info: str


PALETTES: dict[str, Palette] = {
    "inari": Palette("inari", "#ffc21a", "#ffe27a", "#f3ead2", "#9c8b5e", "#3b3420",
                     "#ffd84d", "#ff9f1c", "#ff5f56", "#e8c872"),
    "amber": Palette("amber", "#ffa600", "#ffcc66", "#f6e7cf", "#9a7a48", "#3d2f17",
                     "#ffbf3c", "#ff7b29", "#ff4d4d", "#e0b070"),
    "lemon": Palette("lemon", "#f5e400", "#fff67a", "#f2f2dc", "#8f8a55", "#34321c",
                     "#e3f53b", "#ffb21c", "#ff5f56", "#d8d27a"),
    "kitsune": Palette("kitsune", "#ffb000", "#ff6b35", "#fbe9d7", "#a07a5a", "#3a2a1e",
                       "#ffcf40", "#ff6b35", "#ff3b3b", "#f0a868"),
}

ORDER = list(PALETTES)


def get(name: str) -> Palette:
    return PALETTES.get(name, PALETTES["inari"])


def next_name(name: str) -> str:
    i = ORDER.index(name) if name in ORDER else 0
    return ORDER[(i + 1) % len(ORDER)]


def rich_theme(p: Palette) -> RichTheme:
    return RichTheme({
        "accent": f"bold {p.accent}",
        "accent2": p.accent2,
        "text": p.text,
        "dim": p.dim,
        "good": p.good,
        "warn": p.warn,
        "bad": p.bad,
        "info": p.info,
        "prompt.choices": p.accent,
        "prompt.default": p.dim,
        "repr.number": p.accent2,
        "table.header": f"bold {p.accent}",
        "progress.description": p.text,
        "progress.percentage": p.accent,
        "progress.download": p.accent2,
        "progress.data.speed": p.dim,
        "bar.complete": p.accent,
        "bar.finished": p.good,
        "bar.back": p.track,
    })


def make_console(name: str = "inari") -> Console:
    return Console(theme=rich_theme(get(name)), highlight=False)
