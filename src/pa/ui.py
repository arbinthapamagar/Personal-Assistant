"""Terminal presentation.

Kept apart from the agent so the same loop can be driven by a different front
end later (a daemon, an editor plugin) without untangling print calls.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text as RichText

from .security import Verdict, generalize


class UI:
    def __init__(self, *, plain: bool = False, quiet: bool = False) -> None:
        self.console = Console(no_color=plain, highlight=not plain)
        self.quiet = quiet
        self._streaming = False

    # ---- basics -------------------------------------------------------------

    def banner(self, profile: str, model: str, tools: int, session: str) -> None:
        if self.quiet:
            return
        self.console.print(
            f"[bold]personal assistant[/]  "
            f"[cyan]{profile}[/]:[dim]{model}[/]  "
            f"[dim]{tools} tools  session {session}[/]"
        )
        self.console.print("[dim]/help for commands, /quit to exit[/]\n")

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

    # ---- streaming assistant text -------------------------------------------

    def stream(self, delta: str) -> None:
        """Write a token as it arrives. Deliberately raw - Markdown cannot be
        rendered incrementally without redrawing the whole block."""
        if not self._streaming:
            self._streaming = True
        self.console.file.write(delta)
        self.console.file.flush()

    def end_stream(self) -> None:
        if self._streaming:
            self.console.file.write("\n")
            self.console.file.flush()
            self._streaming = False

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
        "once" | "always" | "skip" | "never". The caller applies it to the gate -
        the UI does not mutate policy itself.
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
        self.console.print(
            f"  [green]y[/] run once   "
            f"[green]a[/] always allow [cyan]{pattern}[/]   "
            f"[red]n[/] skip   "
            f"[red]d[/] always deny [cyan]{pattern}[/]"
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
