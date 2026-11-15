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
from rich.markup import escape

HELP = [("/help", "this list"),
        ("/clear", "start a new conversation (keeps the system prompt)"),
        ("/regen", "answer the last message again"),
        ("/undo", "drop the last exchange"),
        ("/system [TEXT]", "show or set the system prompt (starts a new conversation)"),
        ("/think", "show / hide the model's reasoning"),
        ("/info", "model, server and conversation size"),
        ("/exit", "leave (also /quit, /bye, Ctrl+D, Ctrl+C at an empty prompt)"),
        ('"""', "start / end a multi-line message (or end a line with \\ to continue it)")]


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


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    stopped: bool = False
    finish: str = ""
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


class Chat:
    def __init__(self, console: Console, url: str, *, model: str = "", label: str = "", system: str = "",
                 params: dict | None = None, show_reasoning: bool = True) -> None:
        self.console = console
        self.url = url.rstrip("/")
        self.model = model
        self.label = label or model or url
        self.system = system
        self.params = dict(params or {})
        self.show_reasoning = show_reasoning
        self.history: list[dict] = []

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
        out: queue.Queue = queue.Queue()
        holder: dict = {}
        cancel = threading.Event()
        worker = threading.Thread(target=self._stream, args=(body, out, holder, cancel), daemon=True)
        rep, mode, t0 = Reply(), "", time.monotonic()
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
        rep.seconds = time.monotonic() - t0
        if mode:
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
            try:
                line = self.console.input("[accent]› [/]" if not lines and not block else "[dim]… [/]")
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
        elif cmd == "/info":
            turns = sum(1 for m in self.history if m["role"] == "user")
            chars = sum(len(m["content"]) for m in self.messages())
            self.console.print(f"[dim]  {escape(self.label)} · {self.url} · {turns} message(s) · {chars} chars[/]")
        else:
            self.console.print(f"[warn]  unknown command {escape(cmd)}[/] [dim]— /help[/]")
        return True

    def loop(self) -> None:
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
