"""Terminal presentation.

Kept apart from the agent so the same loop can be driven by a different front
end later (a daemon, an editor plugin) without untangling print calls.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text as RichText

from .security import Verdict, generalize

# A compact wordmark shown at startup. Kept small so it fits an 80-column
# terminal and does not dominate the screen on every launch.
_WORDMARK = r"""
  __ _ _ __| |__ (_)_ __
 / _` | '__| '_ \| | '_ \
| (_| | |  | |_) | | | | |
 \__,_|_|  |_.__/|_|_| |_|   assistant
"""


class UI:
    def __init__(self, *, plain: bool = False, quiet: bool = False) -> None:
        self.console = Console(no_color=plain, highlight=not plain)
        self.quiet = quiet
        self._streaming = False
        self._live = None          # rich.live.Live during a streamed reply
        self._stream_buf = ""      # accumulated text, re-rendered as Markdown

    # ---- basics -------------------------------------------------------------

    def banner(self, profile: str, model: str, tools: int, session: str) -> None:
        """A welcome card: the wordmark, what it is, and how to start."""
        if self.quiet:
            return
        wordmark = RichText(_WORDMARK.strip("\n"), style="bold cyan")
        subtitle = RichText("your local AI assistant", style="dim italic")
        status = RichText.assemble(
            ("model  ", "dim"), (f"{profile}", "bold green"), (" · ", "dim"),
            (f"{model}\n", "green"),
            ("tools  ", "dim"), (f"{tools} ready", "bold"), (" · ", "dim"),
            (f"session {session[:8]}", "dim"),
        )
        hints = RichText.assemble(
            ("type your request and press Enter\n", "white"),
            ("/help", "cyan"), (" commands   ", "dim"),
            ("/profile", "cyan"), (" switch model   ", "dim"),
            ("/voice", "cyan"), (" talk   ", "dim"),
            ("/quit", "cyan"), (" exit", "dim"),
        )
        body = Group(
            Align.center(wordmark),
            Align.center(subtitle),
            RichText(""),
            status,
            RichText(""),
            hints,
        )
        self.console.print(
            Panel(body, border_style="cyan", padding=(1, 3), title="[bold]arbin-assistant[/]",
                  title_align="left")
        )
        self.console.print()

    def info(self, text: str) -> None:
        self.console.print(f"[dim]{text}[/]")

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]![/] {text}")

    def error(self, text: str) -> None:
        self.console.print(f"[red]x[/] {text}")

    def rule(self, text: str = "") -> None:
        self.console.rule(f"[dim]{text}[/]" if text else "")

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(text))

    # ---- working spinner ----------------------------------------------------

    def working(self, label: str = "thinking"):
        """A live spinner shown while the model is loading or thinking, so a
        slow local model reads as busy rather than frozen. Start it before the
        wait, stop it when output begins. A no-op on a non-TTY / quiet, so
        piped output stays clean."""
        return _Spinner(self, label)

    # ---- streaming assistant text -------------------------------------------

    def stream(self, delta: str) -> None:
        """Render tokens as they arrive. On a real terminal this streams as live
        Markdown - fenced code becomes a highlighted, copy-pasteable box as soon
        as its closing ``` arrives - by re-rendering the growing buffer in place.
        Piped/quiet output just writes raw text so scripts stay clean."""
        self._stream_buf += delta
        if self._live is not None:
            self._live.update(Markdown(self._stream_buf))
            return
        if self._live is None and not self._streaming and self.console.is_terminal and not self.quiet:
            from rich.live import Live

            # Live re-renders the whole Markdown each update; throttle refreshes
            # so long replies don't thrash the terminal.
            self._live = Live(Markdown(self._stream_buf), console=self.console,
                              refresh_per_second=12, transient=False)
            self._live.start()
            self._streaming = True
            return
        # Non-TTY / quiet: raw passthrough.
        self._streaming = True
        self.console.file.write(delta)
        self.console.file.flush()

    def end_stream(self) -> None:
        if self._live is not None:
            # If the model wrapped its whole reply in JSON, unwrap it for the
            # final in-place render - Live replaces the streamed JSON frames
            # with clean text, so the user never keeps the raw {"text": ...}.
            from .harness import unwrap_json_answer

            final = unwrap_json_answer(self._stream_buf)
            self._live.update(Markdown(final))
            self._live.stop()
            self._live = None
        elif self._streaming:
            self.console.file.write("\n")
            self.console.file.flush()
        self._streaming = False
        self._stream_buf = ""

    # ---- agent steps --------------------------------------------------------

    def thinking(self, text: str) -> None:
        if self.quiet:
            return
        self.console.print(
            Panel(RichText(text.strip()), title="thinking", border_style="dim", expand=False)
        )

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        rendered = _render_args(args)
        self.console.print(f"[bold cyan]->[/] [bold]{name}[/] {rendered}")

    def tool_result(self, name: str, text: str, is_error: bool) -> None:
        if is_error:
            self.console.print(f"   [red]{text.strip()[:600]}[/]")
            return
        if self.quiet:
            return
        body = text.strip()
        lines = body.splitlines()
        shown = "\n".join(lines[:12])
        more = f"\n[dim]... {len(lines) - 12} more lines[/]" if len(lines) > 12 else ""
        self.console.print(f"[dim]{_indent(shown)}[/]{more}")

    def usage(self, line: str) -> None:
        if not self.quiet:
            self.console.print(f"[dim]{line}[/]")

    # ---- approval prompt ----------------------------------------------------

    def ask_approval(self, key: str, description: str) -> tuple[str, str]:
        """Ask the user about one tool call.

        Returns (decision, pattern) where decision is one of
        "once" | "always" | "skip" | "never". On a real terminal this is an
        arrow-key menu; piped or non-interactive it falls back to y/a/n/d text.
        """
        pattern = generalize(key)
        self.console.print()
        self.console.print(
            Panel(
                RichText.assemble(
                    (description, "bold"),
                    ("\n\nkey: ", "dim"),
                    (key[:200], "cyan"),
                ),
                title="approval needed",
                border_style="yellow",
                expand=False,
            )
        )

        from . import tui

        if tui.interactive():
            choice = tui.select(
                "",
                [
                    tui.Choice("once", "Yes, run it once"),
                    tui.Choice("always", "Yes, and don't ask again", f"for {pattern}"),
                    tui.Choice("skip", "No, skip this"),
                    tui.Choice("never", "No, and never allow this", f"for {pattern}"),
                ],
                footer="↑/↓ move · enter select · esc = skip",
            )
            return (choice or "skip"), pattern

        # Fallback for pipes / non-TTY: the original single-key prompt.
        self.console.print(
            f"  [green]y[/] once   [green]a[/] always [cyan]{pattern}[/]   "
            f"[red]n[/] skip   [red]d[/] never [cyan]{pattern}[/]"
        )
        while True:
            try:
                answer = self.console.input("  [bold]choice[/] [y/a/n/d] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return "skip", pattern
            if answer in ("y", "yes", ""):
                return "once", pattern
            if answer in ("n", "no", "s", "skip"):
                return "skip", pattern
            if answer in ("a", "always"):
                return "always", pattern
            if answer in ("d", "deny"):
                return "never", pattern
            self.console.print("  [dim]answer y, a, n, or d[/]")

    def choose(self, title: str, options: list[tuple[str, str]], *,
               footer: str = "") -> str | None:
        """Generic arrow-key picker. `options` is [(key, label), ...]. Returns
        the chosen key, or None if cancelled / non-interactive."""
        from . import tui

        if not tui.interactive():
            return None
        choices = [tui.Choice(k, label) for k, label in options]
        return tui.select(title, choices, footer=footer or "↑/↓ move · enter select · esc cancel")

    # ---- tables -------------------------------------------------------------

    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        table = Table(title=title, title_style="bold", box=None, pad_edge=False)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def code(self, text: str, lang: str = "yaml") -> None:
        self.console.print(Syntax(text, lang, theme="ansi_dark", background_color="default"))


class _Spinner:
    """Start/stop handle over Rich's Status. Safe (no-op) on a non-TTY."""

    def __init__(self, ui: "UI", label: str) -> None:
        self._ui = ui
        self._label = label
        self._status = None
        self._active = False

    @property
    def _enabled(self) -> bool:
        return not self._ui.quiet and self._ui.console.is_terminal

    def start(self) -> "_Spinner":
        if self._enabled and self._status is None:
            self._status = self._ui.console.status(f"[cyan]{self._label}[/]", spinner="dots")
            self._status.start()
            self._active = True
        return self

    def update(self, label: str) -> None:
        if self._status is not None:
            self._status.update(f"[cyan]{label}[/]")

    def stop(self) -> None:
        if self._status is not None and self._active:
            self._status.stop()
        # Drop the Status so a later start() builds a fresh one (Rich's Status
        # does not cleanly restart after stop()).
        self._status = None
        self._active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def _indent(text: str, prefix: str = "   ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _render_args(args: dict[str, Any]) -> str:
    """Compact one-line argument display, with long values elided."""
    if not args:
        return ""
    bits = []
    for key, value in args.items():
        if isinstance(value, str):
            shown = value if len(value) <= 70 else value[:67] + "..."
            shown = shown.replace("\n", "\\n")
            bits.append(f"{key}={shown!r}")
        else:
            bits.append(f"{key}={json.dumps(value)[:70]}")
    return "[dim]" + "  ".join(bits) + "[/]"
