"""A yes / no confirmation modal shared by the main menu and the setup wizard.

``Delete`` is the destructive button; *Cancel* is focused by default so a stray Enter never deletes.
Keys: ←→ / Tab move between buttons, Enter presses the focused one, y confirms, n / Esc / q cancel.
"""

from __future__ import annotations

from typing import Callable

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from inari import theme as th


class ConfirmModal:
    def __init__(self) -> None:
        self.active = False
        self.title = ""
        self.lines: list[Text] = []
        self.ok_label = "Delete"
        self.on_yes: Callable[[], None] | None = None
        self.focus_ok = False

    def open(self, title: str, lines: list[Text | str], on_yes: Callable[[], None], *, ok_label: str = "Delete") -> None:
        self.active = True
        self.title = title
        self.lines = [ln if isinstance(ln, Text) else Text(ln) for ln in lines]
        self.ok_label = ok_label
        self.on_yes = on_yes
        self.focus_ok = False

    def close(self) -> None:
        self.active = False
        self.on_yes = None

    def handle(self, key: str) -> str:
        """Returns "yes" (on_yes was called), "no" (closed) or "" (still open)."""
        if key in ("left", "right", "tab", "h", "l"):
            self.focus_ok = not self.focus_ok
            return ""
        if key in ("y", "Y") or (key == "enter" and self.focus_ok):
            act = self.on_yes
            self.close()
            if act:
                act()
            return "yes"
        if key in ("n", "N", "esc", "q", "Q", "ctrl+c", "enter"):
            self.close()
            return "no"
        return ""

    def render(self, p: th.Palette) -> RenderableType:
        def button(label: str, on: bool, danger: bool) -> Text:
            if on:
                return Text(f"  {label}  ", style=f"bold {p.bad if danger else p.accent} reverse")
            return Text(f"[ {label} ]", style=p.bad if danger else p.dim)

        buttons = Text.assemble(button("Cancel", not self.focus_ok, False), "    ",
                                button(self.ok_label, self.focus_ok, True))
        hint = Text.assemble(("←→", f"bold {p.accent}"), (" choose   ", p.dim), ("Enter", f"bold {p.accent}"),
                             (" press   ", p.dim), ("y", f"bold {p.accent}"), (f" {self.ok_label.lower()}   ", p.dim),
                             ("Esc", f"bold {p.accent}"), (" cancel", p.dim))
        body = Group(*[ln.copy() for ln in self.lines], Text(""), Align.center(buttons), Text(""),
                     Align.center(hint))
        panel = Panel(body, title=Text(f" {self.title} ", style=f"bold {p.bad}"), title_align="left",
                      border_style=p.bad, box=box.ROUNDED, padding=(1, 3), width=72)
        return Align.center(panel, vertical="middle")
