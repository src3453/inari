"""Colour themes (cycle with `t` in the monitor). kitsune is the default; washi is for light terminals."""

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
    desc: str = ""  # one-line description for the theme picker
    bg: str = "#16161a"  # the terminal background this theme is made for (picker preview only)


PALETTES: dict[str, Palette] = {
    "kitsune": Palette("kitsune", "#ffb000", "#ff6b35", "#fbe9d7", "#a07a5a", "#3a2a1e",
                       "#ffcf40", "#ff6b35", "#ff3b3b", "#f0a868",
                       desc="golden fox — warm amber and vermilion (default)"),
    "sora": Palette("sora", "#4fc3f7", "#9be7ff", "#e3f2fd", "#6b8aa3", "#1c2e3d",
                    "#69f0ae", "#ffb74d", "#ff5f56", "#81d4fa",
                    desc="sky — cool light blue"),
    "matcha": Palette("matcha", "#7ed957", "#c5f29b", "#e8f5e0", "#7a9a6a", "#1f3020",
                      "#b8f36b", "#ffc857", "#ff5f56", "#7fdbca",
                      desc="green tea — fresh greens"),
    "sumi": Palette("sumi", "#ffffff", "#cfcfcf", "#c8c8c8", "#7a7a7a", "#303030",
                    "#ececec", "#a8a8a8", "#ff5f56", "#b8b8b8",
                    desc="ink — greyscale, red only for errors"),
    "washi": Palette("washi", "#d9480f", "#b35c00", "#2b2b2b", "#8a8174", "#e6dfd3",
                     "#2f8f46", "#c77700", "#d32f2f", "#1e6fb8",
                     desc="paper — dark text for light terminal backgrounds", bg="#faf7f0"),
}

DEFAULT = "kitsune"
ORDER = list(PALETTES)


def get(name: str) -> Palette:
    return PALETTES.get(name, PALETTES[DEFAULT])


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


def make_console(name: str = DEFAULT) -> Console:
    return Console(theme=rich_theme(get(name)), highlight=False)
