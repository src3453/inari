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
        self.danger = True

    def open(self, title: str, lines: list[Text | str], on_yes: Callable[[], None], *, ok_label: str = "Delete",
             danger: bool = True) -> None:
        self.active = True
        self.danger = danger
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
                                button(self.ok_label, self.focus_ok, self.danger))
        hint = Text.assemble(("←→", f"bold {p.accent}"), (" choose   ", p.dim), ("Enter", f"bold {p.accent}"),
                             (" press   ", p.dim), ("y", f"bold {p.accent}"), (f" {self.ok_label.lower()}   ", p.dim),
                             ("Esc", f"bold {p.accent}"), (" cancel", p.dim))
        body = Group(*[ln.copy() for ln in self.lines], Text(""), Align.center(buttons), Text(""),
                     Align.center(hint))
        edge = p.bad if self.danger else p.accent
        panel = Panel(body, title=Text(f" {self.title} ", style=f"bold {edge}"), title_align="left",
                      border_style=edge, box=box.ROUNDED, padding=(1, 3), width=72)
        return Align.center(panel, vertical="middle")


class ChoiceModal:
    """Pick one of a few actions: ↑↓ / ←→ / Tab move, Enter or the option's hotkey picks, Esc / q cancels."""

    def __init__(self) -> None:
        self.active = False
        self.title = ""
        self.lines: list[Text] = []
        self.options: list[tuple[str, str, str, str]] = []  # (value, hotkey, label, description)
        self.sel = 0

    def open(self, title: str, lines: list[Text | str], options: list[tuple[str, str, str, str]],
             *, default: str = "") -> None:
        self.active = True
        self.title = title
        self.lines = [ln if isinstance(ln, Text) else Text(ln) for ln in lines]
        self.options = options
        self.sel = next((i for i, o in enumerate(options) if o[0] == default), 0)

    def handle(self, key: str) -> str:
        """Returns the picked value, "no" (closed) or "" (still open)."""
        n = len(self.options)
        if key in ("up", "left", "k"):
            self.sel = (self.sel - 1) % n
            return ""
        if key in ("down", "right", "tab", "j"):
            self.sel = (self.sel + 1) % n
            return ""
        if key == "enter":
            self.active = False
            return self.options[self.sel][0]
        for value, hot, _, _ in self.options:
            if key == hot:
                self.active = False
                return value
        if key in ("esc", "q", "Q", "ctrl+c", "n", "N"):
            self.active = False
            return "no"
        return ""

    def render(self, p: th.Palette) -> RenderableType:
        rows = []
        for i, (_, hot, label, desc) in enumerate(self.options):
            on = i == self.sel
            rows.append(Text.assemble(("▶ " if on else "  ", f"bold {p.accent}"),
                                      (f" {label:<12} ", f"bold {p.accent} reverse" if on else p.accent2),
                                      ("  ", ""), (hot, f"bold {p.accent}"), (f"  {desc}", p.dim)))
        hint = Text.assemble(("↑↓", f"bold {p.accent}"), (" choose   ", p.dim), ("Enter", f"bold {p.accent}"),
                             (" start   ", p.dim), ("Esc", f"bold {p.accent}"), (" cancel", p.dim))
        body = Group(*[ln.copy() for ln in self.lines], *([Text("")] if self.lines else []), *rows, Text(""),
                     Align.center(hint))
        panel = Panel(body, title=Text(f" {self.title} ", style=f"bold {p.accent}"), title_align="left",
                      border_style=p.accent, box=box.ROUNDED, padding=(1, 3), width=72)
        return Align.center(panel, vertical="middle")
