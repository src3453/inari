"""Theme picker shared by the main menu, the monitor and the hub: a short description and a colour
preview for every theme. Moving the cursor previews the theme live; Enter keeps it, Esc reverts."""

from __future__ import annotations

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inari import theme as th


class ThemePicker:
    def __init__(self) -> None:
        self.active = False
        self.sel = 0
        self.orig = th.DEFAULT

    def open(self, current: str) -> None:
        self.active = True
        self.orig = current if current in th.ORDER else th.DEFAULT
        self.sel = th.ORDER.index(self.orig)

    @property
    def name(self) -> str:
        """The theme to show right now (the highlighted one while open)."""
        return th.ORDER[self.sel] if self.active else self.orig

    def handle(self, key: str) -> str:
        """Returns "move" (preview changed), "pick" (Enter), "cancel" (Esc / t / q) or "" (ignored)."""
        n = len(th.ORDER)
        step = {"up": -1, "k": -1, "left": -1, "down": 1, "j": 1, "right": 1, "tab": 1}.get(key)
        if step is not None:
            self.sel = (self.sel + step) % n
            return "move"
        if key.isdigit() and 1 <= int(key) <= n:
            self.sel = int(key) - 1
            return "move"
        if key == "enter":
            self.active = False
            self.orig = th.ORDER[self.sel]
            return "pick"
        if key in ("esc", "t", "q", "Q", "ctrl+c"):
            self.active = False
            self.sel = th.ORDER.index(self.orig)
            return "cancel"
        return ""

    def render(self, p: th.Palette) -> RenderableType:
        """``p`` is the palette the surrounding UI is drawn in (i.e. the one being previewed)."""
        t = Table.grid(padding=(0, 2))
        t.add_column(width=2, no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True, overflow="ellipsis")
        for i, name in enumerate(th.ORDER):
            q = th.PALETTES[name]
            on = i == self.sel
            cur = Text("▶" if on else str(i + 1), style=f"bold {p.accent}" if on else p.dim)
            label = Text(name, style=f"bold {p.accent}" if on else p.text)
            if name == self.orig:
                label.append(" ✓", style=p.good)
            desc = Text(q.desc, style=p.text if on else p.dim)
            t.add_row(cur, label, swatch(q), desc)
            t.add_row("", "", "", "")
        hint = Text.assemble(("↑↓", f"bold {p.accent}"), (" preview   ", p.dim), ("Enter", f"bold {p.accent}"),
                             (" keep   ", p.dim), ("Esc", f"bold {p.accent}"), (" cancel", p.dim))
        body = Table.grid()
        body.add_row(t)
        body.add_row(hint)
        return Panel(body, title=Text(" Theme ", style=f"bold {p.accent}"), title_align="left",
                     border_style=p.accent, box=box.ROUNDED, padding=(1, 2))


def swatch(q: th.Palette) -> Text:
    """A small sample of the theme drawn on the background it is made for."""
    bg = f"on {q.bg}"
    t = Text()
    t.append(" INARI ", style=f"bold {q.accent} {bg}")
    t.append("text ", style=f"{q.text} {bg}")
    t.append("dim ", style=f"{q.dim} {bg}")
    t.append("████", style=f"{q.accent} on {q.track}")
    t.append("    ", style=f"on {q.track}")
    t.append(" ", style=bg)
    for c in (q.accent2, q.good, q.warn, q.bad, q.info):
        t.append("●", style=f"{c} {bg}")
    t.append(" ", style=bg)
    return t
