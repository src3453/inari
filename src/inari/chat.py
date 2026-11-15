"""``inari cli``: chat with a model right in the terminal, like ``llama-cli``.

The model runs in llama-server (started for the chat, or one that is already serving — a profile on its own
or the router), and the chat talks to its OpenAI-compatible ``/v1/chat/completions`` with streaming.
The request runs on a worker thread so Ctrl+C stops a reply immediately, even while the prompt is processed.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.console import Group
from rich.segment import Segment
from rich.text import Text

HELP = [("/help", "this list"),
        ("/clear", "start a new conversation (keeps the system prompt)"),
        ("/regen", "answer the last message again"),
        ("/undo", "drop the last exchange"),
        ("/system [TEXT]", "show or set the system prompt (starts a new conversation)"),
        ("/think", "show / hide the model's reasoning"),
        ("/effort [N]", "thinking budget: tokens, low/medium/high, off, or auto (server default)"),
        ("/md", "Markdown rendering on / off (plain text)"),
        ("/info", "model, server and conversation size"),
        ("/exit", "leave (also /quit, /bye, Ctrl+D, Ctrl+C at an empty prompt)"),
        ('"""', "start / end a multi-line message (or end a line with \\ to continue it)")]

# what the input line offers while a /command is typed: (inserted text, description)
COMMANDS = [("/help", "list the commands"),
            ("/clear", "new conversation + clear the screen"),
            ("/regen", "answer the last message again"),
            ("/undo", "drop the last exchange"),
            ("/system ", "show or set the system prompt"),
            ("/think", "show / hide the model's reasoning"),
            ("/effort ", "thinking budget (tokens, low/medium/high, off, auto)"),
            ("/md", "Markdown rendering on / off"),
            ("/info", "model, server and conversation size"),
            ("/exit", "leave")]
ALIASES = {"/new": "/clear", "/reset": "/clear", "/retry": "/regen", "/markdown": "/md", "/plain": "/md",
           "/budget": "/effort",
           "/quit": "/exit", "/bye": "/exit", "/q": "/exit", "/?": "/help"}


def make_session(console: Console):
    """A prompt_toolkit session with /command suggestions, or None (not a terminal / not installed)."""
    import sys

    if not (console.is_terminal and sys.stdin.isatty()):
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.application import get_app
        from prompt_toolkit.filters import Condition, has_focus
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.layout import Dimension, Float, FloatContainer, HSplit, ScrollOffsets, Window
        from prompt_toolkit.layout.margins import ScrollbarMargin
        from prompt_toolkit.layout.menus import CompletionsMenu, CompletionsMenuControl, MultiColumnCompletionsMenu
        from prompt_toolkit.styles import Style
    except ImportError:
        console.print("[dim]  (no command suggestions: prompt-toolkit is not installed in this environment)[/]")
        return None

    menu_h = len(COMMANDS)  # the most rows the list ever needs (everything, after a bare "/")

    class SuggestSession(PromptSession):
        """PromptSession whose suggestion list never moves the input line: it floats in the free rows below
        the line, or — when the line sits at the bottom of the screen — is stacked above it. The side is
        picked when the list first opens and kept for the rest of that prompt; above, rows the shrinking
        list gives back are kept as blank padding so the line stays put."""

        def __init__(self, *a, **kw) -> None:
            self.menu_above: bool | None = None
            self.menu_peak = 0
            super().__init__(*a, **kw)

        def prompt(self, *a, **kw):
            self.menu_above, self.menu_peak = None, 0
            return super().prompt(*a, **kw)

        def _rows(self) -> int:
            """Rows the list takes right now (0 = closed)."""
            st = self.default_buffer.complete_state
            if get_app().is_done or not st or not st.completions:
                return 0
            return min(len(st.completions), menu_h)

        def _above(self) -> bool:
            if self.menu_above is None and self._rows():
                try:  # rows free under the (single) input line
                    r = get_app().renderer
                    free = r.output.get_size().rows - r.rows_above_layout - 1
                except Exception:  # unknown yet → assume the bottom, which never needs room below
                    free = 0
                self.menu_above = free < menu_h
            return bool(self.menu_above)

        def _above_rows(self) -> int:
            rows = self._rows() if self._above() else 0
            self.menu_peak = max(self.menu_peak, rows)
            return rows

        def _pad_rows(self) -> int:
            rows = self._above_rows()  # first: it may raise the peak
            return 0 if get_app().is_done else self.menu_peak - rows

        def _create_layout(self):
            layout = super()._create_layout()
            focused = has_focus(self.default_buffer)
            stack = [layout.container]
            while stack:
                c = stack.pop()
                if isinstance(c, FloatContainer) and isinstance(c.content, HSplit):
                    c.floats = [f for f in c.floats
                                if not isinstance(f.content, (CompletionsMenu, MultiColumnCompletionsMenu))]
                    c.floats.insert(0, Float(xcursor=True, ycursor=True, transparent=True, content=CompletionsMenu(
                        max_height=menu_h, scroll_offset=1, extra_filter=focused & ~Condition(self._above))))
                    c.content.children[0:0] = [
                        Window(height=lambda: Dimension.exact(self._pad_rows())),
                        Window(CompletionsMenuControl(), height=lambda: Dimension.exact(self._above_rows()),
                               dont_extend_width=True, scroll_offsets=ScrollOffsets(top=1, bottom=1), style="class:completion-menu",
                               right_margins=[ScrollbarMargin(display_arrows=False)]),
                    ]
                    return layout
                stack.extend(c.get_children())
            return layout

    class CommandCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text or "\n" in document.text:
                return
            word = text.lower()
            seen = set()
            for name, desc in COMMANDS:
                if name.startswith(word):
                    seen.add(name.strip())
                    yield Completion(name, start_position=-len(text), display=name.strip(), display_meta=desc)
            for alias, target in ALIASES.items():  # aliases only once they are typed out further than the target
                if alias.startswith(word) and target not in seen and len(word) > 1:
                    yield Completion(alias, start_position=-len(text), display_meta=f"= {target}")

    def hex_of(name: str, fallback: str) -> str:
        try:
            c = console.get_style(name).color
            return c.get_truecolor().hex if c else fallback
        except Exception:
            return fallback

    accent = hex_of("accent", "#e8b830")
    style = Style.from_dict({
        "prompt": f"{accent} bold",
        "cont": "#808080",
        "completion-menu.completion": "bg:#262626 #c8c8c8",
        "completion-menu.completion.current": f"bg:{accent} #000000",
        "completion-menu.meta.completion": "bg:#262626 #808080",
        "completion-menu.meta.completion.current": f"bg:{accent} #000000",
        "scrollbar.background": "bg:#3a3a3a",
        "scrollbar.button": f"bg:{accent}",
    })
    try:
        return SuggestSession(completer=CommandCompleter(), complete_while_typing=True, history=InMemoryHistory(),
                             style=style, reserve_space_for_menu=0)
    except Exception:  # e.g. no Windows console behind stdout (mintty) → plain input
        return None


class ChatError(RuntimeError):
    pass


def tail(path: str | None, n: int = 12) -> str:
    if not path:
        return ""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def wait_ready(console: Console, url: str, *, alive=None, log: str | None = None, label: str = "") -> None:
    """Block until ``url``/health answers 200. ``alive()`` False (the process exited) raises ChatError."""
    t0 = last_ok = time.monotonic()
    with httpx.Client(timeout=2.0) as c, console.status(f"[accent]Starting {escape(label)}…[/]") as st:
        while True:
            state = "starting llama-server"
            try:
                r = c.get(url.rstrip("/") + "/health")
                last_ok = time.monotonic()
                if r.status_code == 200:
                    return
                state = "loading the model" if r.status_code == 503 else f"HTTP {r.status_code}"
            except httpx.HTTPError:
                if alive is None and time.monotonic() - last_ok > 6:  # nothing of ours is starting there
                    raise ChatError(f"cannot reach {url}")
            if alive is not None and not alive():
                msg = "llama-server exited while starting"
                last = tail(log)
                raise ChatError(f"{msg}\n{last}" if last else msg)
            st.update(f"[accent]{state}[/] [dim]{escape(label)} · {time.monotonic() - t0:.0f}s · Ctrl+C to cancel[/]")
            time.sleep(0.4)


def split_stable(text: str) -> tuple[str, str]:
    """Split streamed Markdown into (finished blocks, block still being written).

    The cut goes after the last blank line that is not inside a ``` / ~~~ code fence, so a finished part
    never changes when more text arrives and can be printed for good."""
    cut = offset = 0
    fence = ""
    for line in text.split("\n")[:-1]:  # the last piece is an unfinished line
        s = line.strip()
        if fence:
            if s.startswith(fence) and set(s) == {fence[0]}:
                fence = ""
        elif s.startswith(("```", "~~~")):
            fence = s[:3]
        offset += len(line) + 1
        if not fence and not s:
            cut = offset
    return text[:cut], text[cut:]


class MarkdownStream:
    """Renders a streamed reply as Markdown: finished blocks are printed once, the block being written is
    redrawn in a transient Live area underneath (only its last screenful, so tall code blocks keep up)."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.buf = ""
        self.printed = False
        self.live = Live(console=console, transient=True, auto_refresh=False, vertical_overflow="crop")
        self.live.start()
        self.last = 0.0

    def _md(self, text: str) -> Markdown:
        return Markdown(text, code_theme="monokai", hyperlinks=True)

    def _emit(self, text: str) -> None:
        if not text.strip():
            return
        # Rich pads some blocks (lists, code, tables) with blank lines; trim them so every chunk is
        # separated by exactly one blank line, whatever it starts or ends with
        lines = self.console.render_lines(self._md(text), self.console.options, pad=False)

        def blank(line: list[Segment]) -> bool:
            return not "".join(s.text for s in line).strip()

        while lines and blank(lines[0]):
            lines.pop(0)
        while lines and blank(lines[-1]):
            lines.pop()
        if not lines:
            return
        out = [Text.assemble(*[(s.text, s.style) for s in line if not s.control]) for line in lines]
        if self.printed:
            out.insert(0, Text(""))
        self.console.print(Group(*out), crop=False, overflow="ignore", no_wrap=True)
        self.printed = True

    def _tail(self) -> str:
        lines = self.buf.split("\n")
        room = max(self.console.size.height - 3, 4)
        if len(lines) <= room:
            return self.buf
        opener = ""
        for ln in lines:  # keep the open fence so the visible tail still renders as code
            s = ln.strip()
            if opener and s.startswith(opener[:3]) and set(s) == {opener[0]}:
                opener = ""
            elif not opener and s.startswith(("```", "~~~")):
                opener = s
        tail = lines[-(room - 1):]
        return "\n".join(([opener] if opener else []) + tail)

    def feed(self, piece: str) -> None:
        self.buf += piece
        done, self.buf = split_stable(self.buf)
        if done:
            self.live.update(Text(""), refresh=True)
            self._emit(done)
        now = time.monotonic()
        if done or now - self.last >= 0.08:
            self.live.update(self._md(self._tail()), refresh=True)
            self.last = now

    def close(self) -> None:
        self.live.update(Text(""), refresh=True)
        self.live.stop()
        self._emit(self.buf)
        self.buf = ""


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    stopped: bool = False
    finish: str = ""
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


EFFORT_LEVELS = {"low": 1024, "medium": 4096, "med": 4096, "high": 16384, "max": -2}


def parse_effort(arg: str) -> int | None:
    """``/effort`` argument → thinking budget in tokens. None = server default, 0 = thinking off,
    -2 = unlimited. Accepts ``4096``, ``4k``, ``low/medium/high/max``, ``off``, ``auto``. ValueError if unknown."""
    a = arg.strip().lower().replace(",", "").replace("_", "")
    if a in ("auto", "default", "server", "-1"):
        return None
    if a in ("off", "none", "0", "no"):
        return 0
    if a in ("unlimited", "inf", "max"):
        return -2
    if a in EFFORT_LEVELS:
        return EFFORT_LEVELS[a]
    mult = 1
    if a.endswith("k"):
        a, mult = a[:-1], 1024
    n = int(float(a) * mult)
    if n < 0:
        raise ValueError(arg)
    return n


def describe_effort(budget: int | None) -> str:
    if budget is None:
        return "auto (server default)"
    if budget == 0:
        return "off (no thinking)"
    if budget == -2:
        return "unlimited"
    name = next((k for k, v in EFFORT_LEVELS.items() if v == budget and k != "med"), "")
    return f"{budget:,} tokens" + (f" ({name})" if name else "")


class Chat:
    def __init__(self, console: Console, url: str, *, model: str = "", label: str = "", system: str = "",
                 params: dict | None = None, show_reasoning: bool = True, markdown: bool = True) -> None:
        self.console = console
        self.markdown = markdown  # render replies with Rich Markdown (terminal only; piped output stays raw)
        self.url = url.rstrip("/")
        self.model = model
        self.label = label or model or url
        self.system = system
        self.params = dict(params or {})
        self.show_reasoning = show_reasoning
        self.effort: int | None = None  # thinking budget (see parse_effort); None leaves it to the server
        self.history: list[dict] = []
        self.session = None  # prompt_toolkit input line, set up when the interactive loop starts

    # ───────── requests ─────────
    def messages(self) -> list[dict]:
        return ([{"role": "system", "content": self.system}] if self.system else []) + self.history

    def _stream(self, body: dict, out: queue.Queue, holder: dict, cancel: threading.Event) -> None:
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0, read=None)) as c:
                with c.stream("POST", self.url + "/v1/chat/completions", json=body) as r:
                    holder["response"] = r
                    if r.status_code != 200:
                        raw = r.read().decode("utf-8", "replace")
                        try:
                            err = json.loads(raw).get("error", raw)
                            raw = err.get("message", raw) if isinstance(err, dict) else str(err)
                        except Exception:
                            pass
                        out.put(("error", f"HTTP {r.status_code}: {raw.strip()[:500]}"))
                        return
                    for line in r.iter_lines():
                        if cancel.is_set():
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            out.put(("chunk", json.loads(data)))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:  # connection reset, or the response closed by Ctrl+C
            if not cancel.is_set():
                out.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            out.put(("end", None))

    def ask(self) -> Reply:
        """Send the conversation and stream the answer to the console."""
        body = {"messages": self.messages(), "stream": True, "stream_options": {"include_usage": True},
                **self.params}
        if self.model:
            body["model"] = self.model
        if self.effort == 0:
            kw = dict(body.get("chat_template_kwargs") or {})
            kw["enable_thinking"] = False
            body["chat_template_kwargs"] = kw
            body["thinking_budget_tokens"] = 0
        elif self.effort is not None:
            # llama-server: per-request budget; -1 means "use --reasoning-budget", so unlimited is a huge number
            body["thinking_budget_tokens"] = self.effort if self.effort > 0 else 2**31 - 1
        out: queue.Queue = queue.Queue()
        holder: dict = {}
        cancel = threading.Event()
        worker = threading.Thread(target=self._stream, args=(body, out, holder, cancel), daemon=True)
        rep, mode, t0 = Reply(), "", time.monotonic()
        md: MarkdownStream | None = None
        status = self.console.status("[dim]thinking…  Ctrl+C stops[/]")
        status.start()
        waiting = True
        worker.start()
        try:
            while True:
                try:
                    kind, val = out.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "end":
                    break
                if kind == "error":
                    raise ChatError(val)
                if val.get("timings"):
                    rep.timings = val["timings"]
                if val.get("usage"):
                    rep.usage = val["usage"]
                for ch in val.get("choices") or []:
                    d = ch.get("delta") or {}
                    rc, ct = d.get("reasoning_content"), d.get("content")
                    if ch.get("finish_reason"):
                        rep.finish = ch["finish_reason"]
                    for piece, kind_ in ((rc, "think"), (ct, "say")):
                        if not piece:
                            continue
                        if waiting:
                            status.stop()
                            waiting = False
                        if kind_ == "think":
                            rep.reasoning += piece
                            if not self.show_reasoning:
                                if mode != "think" and self.console.is_terminal:
                                    self.console.print("[dim]· thinking…[/]")
                                    mode = "think"
                                continue
                            if mode != "think":
                                self.console.print("[dim]┄ thinking ┄[/]")
                                mode = "think"
                            self.console.print(piece, end="", style="dim italic", markup=False, highlight=False,
                                               soft_wrap=True)
                        else:
                            if mode == "think" and self.show_reasoning:
                                self.console.print("\n[dim]┄┄┄┄┄┄┄┄┄┄┄┄[/]")
                            mode = "say"
                            rep.content += piece
                            if md is None and self.markdown and self.console.is_terminal:
                                md = MarkdownStream(self.console)
                            if md is not None:
                                md.feed(piece)
                            else:
                                self.console.print(piece, end="", style="text", markup=False, highlight=False,
                                                   soft_wrap=True)
        except KeyboardInterrupt:
            rep.stopped = True
            cancel.set()
            r = holder.get("response")
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
        finally:
            if waiting:
                status.stop()
            if md is not None:
                md.close()
        rep.seconds = time.monotonic() - t0
        if mode and md is None:
            self.console.print()
        return rep

    def stats(self, rep: Reply) -> str:
        t, u = rep.timings, rep.usage
        n = t.get("predicted_n") or u.get("completion_tokens") or 0
        parts = []
        if n:
            tps = t.get("predicted_per_second") or (n / rep.seconds if rep.seconds else 0)
            parts.append(f"{n} tok · {tps:.1f} tok/s")
        pn = t.get("prompt_n") if t else u.get("prompt_tokens")
        if pn:
            pps = t.get("prompt_per_second")
            parts.append(f"prompt {pn} tok" + (f" @ {pps:.0f} tok/s" if pps else ""))
        parts.append(f"{rep.seconds:.1f}s")
        if rep.stopped:
            parts.append("stopped")
        elif rep.finish == "length":
            parts.append("cut off (max tokens)")
        return " · ".join(parts)

    def turn(self) -> None:
        """Answer the conversation as it stands (last message is the user's)."""
        try:
            rep = self.ask()
        except ChatError as e:
            self.history.pop()  # let the user send it again
            self.console.print(f"[bad]✖[/] {escape(str(e))}")
            return
        if rep.content or rep.reasoning:
            self.history.append({"role": "assistant", "content": rep.content})
        else:
            self.history.pop()
        self.console.print(f"[dim]  {self.stats(rep)}[/]")

    # ───────── the loop ─────────
    def read(self) -> str | None:
        """One message from the user. None = leave."""
        lines: list[str] = []
        block = False
        while True:
            first = not lines and not block
            try:
                if self.session is not None:
                    line = self.session.prompt([("class:prompt", "› ")] if first else [("class:cont", "… ")],
                                               complete_while_typing=first)
                else:
                    line = self.console.input("[accent]› [/]" if first else "[dim]… [/]")
            except EOFError:
                return None
            except KeyboardInterrupt:
                if lines:
                    self.console.print("[dim]  (discarded)[/]")
                    lines, block = [], False
                    continue
                return None
            if line.strip() == '"""':
                if block:
                    return "\n".join(lines)
                block = True
                continue
            if block:
                lines.append(line)
                continue
            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            lines.append(line)
            return "\n".join(lines)

    def intro(self) -> None:
        self.console.print(f"[accent]◆[/] chatting with [accent]{escape(self.label)}[/] [dim]at {self.url}[/]")
        self.console.print("[dim]  /help for commands · /exit or Ctrl+D to leave · Ctrl+C stops a reply[/]")
        if self.system:
            self.console.print(f"[dim]  system: {escape(self.system)}[/]")

    def command(self, text: str) -> bool:
        """Handle a /command. False = leave."""
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd in ("/exit", "/quit", "/bye", "/q"):
            return False
        if cmd in ("/help", "/?"):
            for k, d in HELP:
                self.console.print(f"  [accent]{escape(k):<16}[/] [dim]{escape(d)}[/]")
        elif cmd in ("/clear", "/new", "/reset"):
            self.history.clear()
            if self.console.is_terminal:
                self.console.clear()
                self.intro()
            self.console.print("[dim]  new conversation[/]")
        elif cmd in ("/regen", "/retry"):
            if self.history and self.history[-1]["role"] == "assistant":
                self.history.pop()
            if not self.history:
                self.console.print("[dim]  nothing to answer yet[/]")
            else:
                self.turn()
        elif cmd == "/undo":
            while self.history and self.history[-1]["role"] == "assistant":
                self.history.pop()
            if self.history:
                self.history.pop()
                self.console.print("[dim]  dropped the last exchange[/]")
            else:
                self.console.print("[dim]  nothing to undo[/]")
        elif cmd == "/system":
            if arg.strip():
                self.system = arg.strip()
                self.history.clear()
                self.console.print("[dim]  system prompt set · new conversation[/]")
            else:
                self.console.print(f"[dim]  system: {escape(self.system) if self.system else '(none)'}[/]")
        elif cmd == "/think":
            self.show_reasoning = not self.show_reasoning
            self.console.print(f"[dim]  reasoning {'shown' if self.show_reasoning else 'hidden'}[/]")
        elif cmd in ("/effort", "/budget"):
            if arg.strip():
                try:
                    self.effort = parse_effort(arg)
                except ValueError:
                    self.console.print(f"[warn]  /effort: unknown value {escape(arg.strip())}[/] "
                                       "[dim]— a token count (e.g. 2048, 8k), low/medium/high/max, off or auto[/]")
                    return True
                self.console.print(f"[dim]  thinking budget: {escape(describe_effort(self.effort))}[/]")
            else:
                self.console.print(f"[dim]  thinking budget: {escape(describe_effort(self.effort))} · "
                                   "set with /effort N | low | medium | high | max | off | auto[/]")
        elif cmd in ("/md", "/markdown", "/plain"):
            self.markdown = cmd != "/plain" and not self.markdown
            self.console.print(f"[dim]  replies {'rendered as Markdown' if self.markdown else 'shown as plain text'}[/]")
        elif cmd == "/info":
            turns = sum(1 for m in self.history if m["role"] == "user")
            chars = sum(len(m["content"]) for m in self.messages())
            self.console.print(f"[dim]  {escape(self.label)} · {self.url} · {turns} message(s) · {chars} chars · "
                               f"thinking {escape(describe_effort(self.effort))}[/]")
        else:
            self.console.print(f"[warn]  unknown command {escape(cmd)}[/] [dim]— /help[/]")
        return True

    def loop(self) -> None:
        self.session = make_session(self.console)
        self.intro()
        while True:
            text = self.read()
            if text is None:
                self.console.print()
                return
            if not text.strip():
                continue
            if text.startswith("/") and "\n" not in text:
                if not self.command(text.strip()):
                    return
                continue
            self.history.append({"role": "user", "content": text})
            self.turn()

    def once(self, prompt: str) -> bool:
        """Non-interactive: answer ``prompt`` and return."""
        self.history.append({"role": "user", "content": prompt})
        try:
            rep = self.ask()
        except ChatError as e:
            self.console.print(f"[bad]✖[/] {escape(str(e))}")
            return False
        return not rep.stopped
