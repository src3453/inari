"""Non-blocking keyboard reader (Windows: msvcrt, POSIX: cbreak + select)."""

from __future__ import annotations

import os
import queue
import sys
import threading

_WIN_SPECIAL = {"H": "up", "P": "down", "K": "left", "M": "right", "I": "pgup", "Q": "pgdn",
                "G": "home", "O": "end"}
_ANSI = {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "[5~": "pgup", "[6~": "pgdn",
         "[H": "home", "[F": "end"}


def _name(ch: str) -> str:
    return {"\r": "enter", "\n": "enter", "\x1b": "esc", "\x08": "backspace", "\x7f": "backspace",
            "\t": "tab", "\x03": "ctrl+c"}.get(ch, ch)


class KeyReader:
    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._old = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "KeyReader":
        if os.name != "nt" and sys.stdin.isatty():
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        if os.name == "nt":
            self._run_win()
        else:
            self._run_posix()

    def _run_win(self) -> None:
        import msvcrt
        import time

        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.02)
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                self.q.put(_WIN_SPECIAL.get(msvcrt.getwch(), "?"))
            else:
                self.q.put(_name(ch))

    def _run_posix(self) -> None:
        import select

        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue
            data = os.read(fd, 64).decode(errors="ignore")
            i = 0
            while i < len(data):
                if data[i] == "\x1b" and i + 1 < len(data):
                    for seq, name in _ANSI.items():
                        if data.startswith(seq, i + 1):
                            self.q.put(name)
                            i += 1 + len(seq)
                            break
                    else:
                        self.q.put("esc")
                        i += 1
                    continue
                self.q.put(_name(data[i]))
                i += 1
