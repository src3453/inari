"""The inari logo (shape originally made with Rich Art Studio), coloured from the active theme."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from inari import __version__
from inari import theme as th

ART = [
    "▜▛        ▗▖",
    "▐▌▐▛▌▀█▐▛▌▄▖",
    "▟▙▐▌▌█▟▐▌ ▟▙",
]
# for outputs whose encoding lacks block/box glyphs (e.g. cp932 when piped to a file)
ART_FALLBACKS = [
    ["╦╔╗╔╔═╗╦═╗╦", "║║║║╠═╣╠╦╝║", "╩╝╚╝╩ ╩╩╚═╩"],
    ["", "  inari", ""],
]
ASCII_LOGO = ART_FALLBACKS[-1]
ascii_only = False  # set by `inari --nologo`
TAGLINE = ["", "llama.cpp, served golden", f"v{__version__}"]


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def _encodable(rows: list[str], console: Console) -> bool:
    try:
        "".join(rows).encode(console.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def logo_lines(p: th.Palette, console: Console | None = None) -> list[Text]:
    """Logo rows, with a left→right gradient from the theme's accent to its warn colour."""
    if ascii_only:
        return [Text(r, style=f"bold {p.accent}") for r in ASCII_LOGO]
    if console is not None and not _encodable(ART, console):
        rows = next((f for f in ART_FALLBACKS if _encodable(f, console)), ART_FALLBACKS[-1])
        return [Text(r, style=f"bold {p.accent}") for r in rows]
    lines = []
    for row in ART:
        t = Text(no_wrap=True, overflow="crop")
        n = max(len(row) - 1, 1)
        for i, ch in enumerate(row):
            t.append(ch, style=_mix(p.accent, p.warn, i / n))
        lines.append(t)
    return lines


def banner(p: th.Palette, console: Console | None = None) -> Table:
    """Logo with the tagline and version beside it."""
    g = Table.grid(padding=(0, 3))
    g.add_column(no_wrap=True)
    g.add_column(no_wrap=True)
    for art, side in zip(logo_lines(p, console), TAGLINE):
        g.add_row(Text("  ") + art, Text(side, style=p.dim))
    return g


if __name__ == "__main__":
    c = Console()
    for name in th.ORDER:
        c.print(banner(th.get(name), c))
        c.print()
