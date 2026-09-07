"""Interactive terminal widgets, in the style of a modern coding agent.

An inline arrow-key selector for approvals and pickers, and a slash-command
completer for the prompt. Everything here needs a real TTY; callers check
`interactive()` and fall back to plain text prompts when piped, in a one-shot,
or under the daemon. Kept in its own module so the plain UI has no hard
dependency on prompt_toolkit's application machinery.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


def interactive() -> bool:
    """True only when we can drive a full-screen-ish control - a real terminal
    on both ends. Anywhere else, callers use the text fallback."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):
        return False


@dataclass
class Choice:
    key: str          # the value returned when this row is picked
    label: str        # the main line
    hint: str = ""    # dim trailing hint


def select(
    title: str,
    choices: list[Choice],
    *,
    footer: str = "↑/↓ move · enter select · esc cancel",
    default: int = 0,
) -> str | None:
    """Show an inline arrow-key menu and return the chosen key (or None).

    Renders in place (a few lines), not a full-screen popup, so it reads like
    the rest of the conversation - the way an approval prompt should.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import to_formatted_text
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    index = [max(0, min(default, len(choices) - 1))]

    def render():
        lines: list = []
        if title:
            lines.append(("bold", title + "\n"))
        for i, choice in enumerate(choices):
            selected = i == index[0]
            pointer = "❯ " if selected else "  "
            style = "class:sel" if selected else ""
            lines.append((style, f"{pointer}{choice.label}"))
            if choice.hint:
                lines.append(("class:hint", f"  {choice.hint}"))
            lines.append(("", "\n"))
        lines.append(("class:footer", footer))
        return to_formatted_text(lines)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    @kb.add("k")
    def _up(event):
        index[0] = (index[0] - 1) % len(choices)

    @kb.add("down")
    @kb.add("c-n")
    @kb.add("j")
    def _down(event):
        index[0] = (index[0] + 1) % len(choices)

    @kb.add("enter")
    def _accept(event):
        event.app.exit(result=choices[index[0]].key)

    @kb.add("escape")
    @kb.add("c-c")
    @kb.add("q")
    def _cancel(event):
        event.app.exit(result=None)

    # Number keys jump straight to a row (1-9), like a menu.
    for n in range(1, min(len(choices), 9) + 1):
        @kb.add(str(n))
        def _pick(event, n=n):
            event.app.exit(result=choices[n - 1].key)

    from prompt_toolkit.styles import Style

    style = Style.from_dict({
        "sel": "reverse",
        "hint": "#888888",
        "footer": "#888888 italic",
    })

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(render), always_hide_cursor=True)])),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    return app.run()


class SlashCompleter:
    """Completes slash commands as the user types `/`."""

    def __init__(self, commands: dict[str, str]):
        # {name: description}
        self.commands = commands

    def build(self):
        from prompt_toolkit.completion import Completer, Completion

        commands = self.commands

        class _C(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if not text.startswith("/"):
                    return
                word = text[1:]
                for name, desc in commands.items():
                    if name.startswith(word):
                        yield Completion(
                            name,
                            start_position=-len(word),
                            display=f"/{name}",
                            display_meta=desc,
                        )

        return _C()
