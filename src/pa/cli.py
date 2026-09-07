"""Command-line entry point: argument parsing, the REPL, and slash commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, capabilities, config as config_mod, paths, providers, store as store_mod
from .agent import Agent
from .errors import PAError
from .security import Gate
from .session import Session
from .tools.base import ToolContext, build_registry
from .ui import UI

HELP = """\
commands
  /help                 this list
  /quit  /exit          leave (ctrl-d also works)
  /profile [name]       show or switch provider profile
  /model  [name]        show or change the model for this profile
  /models               list models the current provider offers
  /tools                list active tools
  /skills               list loadable skill playbooks
  /tasks                list background tasks
  /memory               memory store stats
  /caps                 what this machine can do
  /harness              show how tools are being called (native/prompted)
  /check                verify the current provider's credentials
  /mode [ask|allow|deny] show or change the permission mode
  /yolo                 skip all permission prompts (toggle)
  /auto <goal>          autonomous mode: pursue a goal across many turns
  /audit                what has been approved or denied this session
  /stats                token usage for this session
  /voice                toggle spoken input and replies
  /say <text>           speak something aloud now
  /new                  start a fresh conversation
  /sessions             recent sessions
  /config               path to the config file (creates a starter if absent)
  /cwd [path]           show or change the agent's working directory
"""


def _slash_help_map() -> dict[str, str]:
    """Parse the HELP block into {command: description} for autocompletion.

    Derived from the one HELP string so the completer never drifts from the
    /help text - a single source of truth for the command list."""
    out: dict[str, str] = {}
    for line in HELP.splitlines():
        line = line.strip()
        if not line.startswith("/"):
            continue
        # e.g. "/profile [name]       show or switch provider profile"
        head, _, desc = line.partition("  ")
        for token in head.split():
            if token.startswith("/"):
                out[token[1:]] = desc.strip()
    return out


def _indent_help(help_text: str) -> str:
    """The REPL HELP block, minus its own 'commands' header, for the --help
    epilog (argparse adds section headers itself)."""
    lines = help_text.strip().splitlines()
    if lines and lines[0].strip() == "commands":
        lines = lines[1:]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    # The epilog carries the in-session slash commands and a few examples, so
    # `--help` is a single complete reference rather than only the flags.
    epilog = f"""
in-session commands (type these at the > prompt)
{_indent_help(HELP)}
examples
  arbin-assistant                         start chatting (uses the default profile)
  arbin-assistant -p gemini-pro           use the smartest Gemini model
  arbin-assistant -p gemini-lite "hi"     one-shot on the fast model
  arbin-assistant -p local                offline, no key
  arbin-assistant --voice                 talk to it out loud
  arbin-assistant --serve                 run the always-on daemon
  arbin-assistant --doctor                show what this machine can do

profiles are defined in ~/.config/arbin-assistant/config.yaml
switch model live with  /profile <name>  or  /model <id>
"""
    parser = argparse.ArgumentParser(
        prog="arbin-assistant",
        description="arbin-assistant - your local AI assistant for the terminal, browser, and desktop.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="Run one prompt and exit.")
    parser.add_argument("-p", "--profile", help="Profile to use (see /profile).")
    parser.add_argument("-m", "--model", help="Override the profile's model.")
    parser.add_argument("--mode", choices=["ask", "allow", "deny"], help="Permission mode.")
    parser.add_argument("--config", type=Path, help="Path to an alternate config file.")
    parser.add_argument("--resume", action="store_true", help="Continue the newest session.")
    parser.add_argument("--session", help="Resume a specific session id.")
    parser.add_argument("--tools", help="Comma-separated tool groups to enable.")
    parser.add_argument("--no-stream", action="store_true", help="Wait for whole responses.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print final answers.")
    parser.add_argument("--plain", action="store_true", help="Disable colour.")
    parser.add_argument("--doctor", action="store_true", help="Report the environment and exit.")
    parser.add_argument("--warm", action="store_true",
                        help="Pre-download the local embedding model, then exit.")
    parser.add_argument("--voice", action="store_true",
                        help="Talk to it: spoken input, spoken replies.")
    parser.add_argument("--yolo", action="store_true",
                        help="Skip ALL permission prompts this session (like --mode allow).")
    parser.add_argument("--auto", action="store_true",
                        help="Autonomous mode: pursue the given prompt as a goal, unattended.")

    daemon_group = parser.add_argument_group("daemon (a persistent agent reachable from anywhere)")
    daemon_group.add_argument("--serve", action="store_true",
                              help="Run the persistent daemon in the foreground.")
    daemon_group.add_argument("--daemon", action="store_true",
                              help="Send this prompt (or REPL) to the running daemon.")
    daemon_group.add_argument("--daemon-status", action="store_true",
                              help="Show the running daemon's status.")
    daemon_group.add_argument("--daemon-stop", action="store_true",
                              help="Ask the running daemon to shut down.")
    daemon_group.add_argument("--install-service", action="store_true",
                              help="Install and start a systemd user service, then exit.")
    daemon_group.add_argument("--uninstall-service", action="store_true",
                              help="Stop and remove the systemd user service, then exit.")

    parser.add_argument("--version", action="version", version=f"arbin-assistant {__version__}")
    return parser


def _config_sets_tools(path) -> bool:
    """Did the user pin a tool list in their config file? If so, respect it
    rather than swapping in the local preset."""
    import yaml as _yaml

    target = path or paths.config_file()
    try:
        raw = _yaml.safe_load(target.read_text()) or {}
    except (OSError, _yaml.YAMLError):
        return False
    return isinstance(raw, dict) and "tools" in raw


def make_approver(ui: UI, gate: Gate):
    """Bridge the UI's four-way answer to the gate's boolean question."""

    def approver(key: str, description: str) -> bool:
        decision, pattern = ui.ask_approval(key, description)
        if decision == "always":
            gate.remember_allow(pattern)
            ui.info(f"allowing {pattern} for this session")
            return True
        if decision == "never":
            gate.remember_deny(pattern)
            ui.info(f"denying {pattern} for this session")
            return False
        return decision == "once"

    return approver


def doctor(ui: UI, cfg: config_mod.Config) -> int:
    caps = capabilities.probe()
    ui.rule("machine")
    ui.console.print(caps.summary())
    ui.rule("config")
    ui.console.print(
        f"config file  {paths.config_file()} "
        f"{'(exists)' if paths.config_file().exists() else '(not created - defaults in use)'}\n"
        f"data dir     {paths.data_dir()}\n"
        f"profiles     {', '.join(sorted(cfg.profiles))}\n"
        f"active       {cfg.active_profile} -> {cfg.profile.provider}:{cfg.profile.model}\n"
        f"mode         {cfg.security.mode}"
    )
    ui.rule("providers")
    for name in sorted(cfg.profiles):
        profile = cfg.profiles[name]
        key = profile.resolve_key()
        if profile.provider in ("ollama",):
            state = "no key needed"
        elif key:
            state = f"key found (${profile.api_key_env})"
        elif profile.provider == "anthropic":
            state = f"no ${profile.api_key_env} - an `ant auth login` profile may still work"
        else:
            state = f"[red]no key[/] - set ${profile.api_key_env}"
        marker = "*" if name == cfg.active_profile else " "
        ui.console.print(f" {marker} {name:8} {profile.provider:14} {profile.model:28} {state}")
    ui.rule("tools")
    try:
        gate = Gate(cfg.security)
        ctx = ToolContext(cfg, caps, gate, lambda k, d: False)
        registry = build_registry(cfg, caps)
        active = {s.name for s in registry.specs(ctx)}
        for tool in registry:
            ok = tool.name in active
            reason = "" if ok else f"  [dim]{tool.available(ctx)[1]}[/]"
            ui.console.print(f"   {'[green]on [/]' if ok else '[dim]off[/]'} {tool.name}{reason}")
    except PAError as exc:
        ui.error(str(exc))
        return 1
    return 0


def _render_steps(agent: Agent, ui: UI, steps, *, streaming: bool) -> bool:
    """Render a stream of agent Steps with a busy-spinner. Shared by a normal
    turn and autonomous mode. Returns False if any step errored."""
    failed = False
    spinner = ui.working("thinking").start()
    stopped = {"v": False}

    def _stop_spin():
        if not stopped["v"]:
            spinner.stop()
            stopped["v"] = True

    try:
        for step in steps:
            _stop_spin()
            if step.kind in ("tool", "result", "error", "thinking"):
                ui.end_stream()
            if step.kind == "text":
                if not streaming:      # streamed text already rendered live
                    ui.markdown(step.text)
            elif step.kind == "thinking":
                ui.thinking(step.text)
            elif step.kind == "tool":
                ui.tool_call(step.tool, step.args)
                spinner.update("working"); stopped["v"] = False; spinner.start()
            elif step.kind == "result":
                _stop_spin()
                ui.tool_result(step.tool, step.text, step.is_error)
            elif step.kind == "usage":
                ui.end_stream()
                ui.usage(step.text)
            elif step.kind == "error":
                failed = True
                ui.error(step.text)
    except KeyboardInterrupt:
        _stop_spin(); ui.end_stream()
        ui.warn("interrupted - the conversation is intact, ask something else")
        return False
    finally:
        _stop_spin(); ui.end_stream()
    return not failed


def run_turn(agent: Agent, ui: UI, prompt: str, *, speak=None) -> bool:
    """Drive one user turn and render it. Returns False if the turn failed."""
    streaming = agent.config.stream and speak is None

    def _on_text(delta: str) -> None:
        ui.stream(delta)

    on_text = _on_text if streaming else None
    ok = _render_steps(agent, ui, agent.turn(prompt, on_text=on_text), streaming=streaming)
    if speak is not None and ok:
        for message in reversed(agent.session.messages):
            if message.role == "assistant" and message.text.strip():
                speak(message.text)
                break
    return ok


def run_autopilot(agent: Agent, ui: UI, goal: str) -> bool:
    """Run the agent autonomously toward a goal, rendering each turn."""
    ui.rule("autopilot")
    ui.info(f"goal: {goal}")
    streaming = agent.config.stream
    on_text = ui.stream if streaming else None
    ok = _render_steps(agent, ui, agent.autopilot(goal, on_text=on_text), streaming=streaming)
    ui.rule()
    return ok


def handle_command(line: str, state: dict) -> bool:
    """Run a slash command. Returns False to exit the REPL."""
    ui: UI = state["ui"]
    agent: Agent = state["agent"]
    cfg: config_mod.Config = state["config"]
    parts = line[1:].split()
    cmd, args = (parts[0] if parts else ""), parts[1:]

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        ui.console.print(HELP)
    elif cmd == "profile":
        name = args[0] if args else None
        if name is None:
            # Arrow-key picker on a real terminal; a listing otherwise.
            options = [
                (n, f"{n:12} {p.provider}:{p.model}"
                    + ("  (current)" if n == cfg.active_profile else ""))
                for n, p in sorted(cfg.profiles.items())
            ]
            name = ui.choose("switch model profile", options,
                             footer="↑/↓ move · enter select · esc cancel")
            if name is None:
                rows = [
                    [("* " if n == cfg.active_profile else "  ") + n, p.provider, p.model]
                    for n, p in sorted(cfg.profiles.items())
                ]
                ui.table("profiles", ["name", "provider", "model"], rows)
        if name is not None:
            if name not in cfg.profiles:
                ui.error(f"no profile {name!r}. Known: {', '.join(sorted(cfg.profiles))}")
            else:
                try:
                    new_cfg = cfg.with_profile(name)
                    provider = providers.build(new_cfg.profile, new_cfg)
                except PAError as exc:
                    ui.error(f"could not switch to {name}: {exc}")
                else:
                    agent.switch_profile(name, provider)
                    state["config"] = new_cfg
                    cfg = new_cfg
                    # Persist so the next launch starts on this profile.
                    saved = config_mod.persist_selection(name)
                    note = "  (saved as default)" if saved else ""
                    ui.info(f"now using {name} -> {provider.model}{note}")
    elif cmd == "model":
        target = args[0] if args else None
        if target is None:
            models = agent.provider.list_models()
            if models:
                target = ui.choose(
                    f"pick a model for {cfg.active_profile}",
                    [(m, m + ("  (current)" if m == agent.provider.model else ""))
                     for m in models],
                )
            if target is None:
                ui.info(f"{cfg.active_profile} -> {agent.provider.model}"
                        + (f"   ({len(models)} available - /models)" if models else ""))
        if target is not None:
            cfg.profile.model = target
            try:
                provider = providers.build(cfg.profile, cfg)
            except PAError as exc:
                ui.error(str(exc))
            else:
                agent.switch_profile(cfg.active_profile, provider)
                saved = config_mod.persist_selection(cfg.active_profile, model=target)
                note = "  (saved as default)" if saved else ""
                ui.info(f"model is now {target}{note}")
    elif cmd == "models":
        models = agent.provider.list_models()
        ui.console.print("\n".join(f"  {m}" for m in models) if models
                         else "[dim]this provider offers no model listing[/]")
    elif cmd == "tools":
        active = {s.name for s in agent.registry.specs(agent.ctx)}
        for tool in agent.registry:
            mark = "[green]on [/]" if tool.name in active else "[dim]off[/]"
            ui.console.print(f"  {mark} [bold]{tool.name}[/] [dim]{tool.description[:80]}[/]")
    elif cmd == "skills":
        if not len(agent.skills):
            ui.info("no skills installed")
        else:
            for skill in agent.skills:
                ui.console.print(
                    f"  [bold]{skill.name}[/] [dim]({skill.origin})[/]  {skill.description}"
                )
    elif cmd == "tasks":
        if agent.ctx.store is None:
            ui.info("no state backend - background tasks disabled")
        else:
            from .tasks import TaskManager

            tasks = TaskManager(agent.ctx.store).list()
            ui.console.print("\n".join(t.describe() for t in tasks) if tasks
                             else "no background tasks")
    elif cmd == "memory":
        from .memory import Memory

        memory = agent.ctx.state.get("memory") or Memory(
            auto_install=cfg.auto_install_deps
        )
        if not memory.available():
            ui.info("memory backend (chromadb) not installed yet - it installs on first use")
        else:
            ui.console.print(memory.stats())
    elif cmd == "voice":
        state["voice_mode"] = not state.get("voice_mode", False)
        if state["voice_mode"]:
            try:
                ui.info(_voice(state).check())
                ui.info("voice mode ON - speak your turns; /voice again to stop")
            except Exception as exc:  # noqa: BLE001
                state["voice_mode"] = False
                ui.error(f"could not start voice: {exc}")
        else:
            ui.info("voice mode OFF - back to typing")
    elif cmd == "say":
        text = " ".join(args)
        if not text:
            ui.warn("usage: /say <text>")
        else:
            _speaker(state)(text)
    elif cmd == "harness":
        ui.info(
            f"harness: {agent._resolved_mode}"
            + (f" (mode '{agent.harness_mode}')" if agent.harness_mode != agent._resolved_mode else "")
        )
        if agent._resolved_mode == "prompted":
            ui.info("tools are described in the prompt and parsed from text - "
                    "the robust path for local models")
    elif cmd == "caps":
        ui.console.print(agent.caps.summary())
        backend = agent.ctx.store.name if agent.ctx.store is not None else "none"
        ui.console.print(f"state       {backend}")
        if agent.ctx.gate.disabled_rules:
            ui.warn(
                "denial-floor rules disabled by config: "
                + ", ".join(agent.ctx.gate.disabled_rules)
            )
    elif cmd == "check":
        ui.info("checking ...")
        ui.console.print(f"  {agent.provider.check()}")
    elif cmd == "mode":
        target = args[0] if args else ui.choose(
            "permission mode",
            [("ask", "ask     - confirm each tool call (safe default)"),
             ("allow", "allow   - run without asking (floor rules still apply)"),
             ("deny", "deny    - refuse anything not explicitly allowed")],
        )
        if target is None:
            ui.info(f"permission mode: {cfg.security.mode}")
        elif target not in ("ask", "allow", "deny"):
            ui.error("mode must be ask, allow, or deny")
        else:
            cfg.security.mode = target
            if target == "allow":
                ui.warn("every tool call will now run without asking - hard denials still apply")
            ui.info(f"permission mode: {target}")
    elif cmd == "yolo":
        if cfg.security.mode == "allow":
            cfg.security.mode = "ask"
            ui.info("permission prompts are back ON")
        else:
            cfg.security.mode = "allow"
            ui.warn("YOLO: every tool call now runs without asking. "
                    + ("The safety floor is OFF too." if agent.ctx.gate.floor_off
                       else "The catastrophic-command floor still applies."))
    elif cmd == "auto":
        goal = " ".join(args)
        if not goal:
            ui.warn("usage: /auto <goal>  — e.g. /auto find and summarise the TODOs in this repo")
        else:
            run_autopilot(agent, ui, goal)
    elif cmd == "audit":
        ui.console.print(agent.ctx.gate.audit())
    elif cmd == "stats":
        ui.console.print(agent.stats())
    elif cmd == "new":
        agent.session = Session(profile=cfg.active_profile, model=agent.provider.model)
        ui.info(f"new session {agent.session.id}")
    elif cmd == "sessions":
        rows = [[sid, when, str(count)] for sid, when, count in Session.listing()]
        ui.table("sessions", ["id", "started", "messages"], rows) if rows else ui.info("none yet")
    elif cmd == "config":
        path = paths.config_file()
        if not path.exists():
            config_mod.write_example(path)
            ui.info(f"created {path}")
        else:
            ui.info(str(path))
    elif cmd == "cwd":
        if args:
            target = Path(args[0]).expanduser()
            if not target.is_dir():
                ui.error(f"not a directory: {target}")
            else:
                agent.ctx.cwd = target.resolve()
                ui.info(f"cwd is now {agent.ctx.cwd}")
        else:
            ui.info(f"cwd {agent.ctx.cwd}   workspace {agent.ctx.workspace}")
    else:
        ui.error(f"unknown command /{cmd} - try /help")
    return True


def repl(agent: Agent, ui: UI, cfg: config_mod.Config, *, voice_mode: bool = False) -> int:
    state = {"ui": ui, "agent": agent, "config": cfg, "voice_mode": voice_mode}
    prompt_session = None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        from .tui import SlashCompleter

        paths.ensure_dirs()
        # Autocomplete slash commands as the user types "/", and show live
        # model/mode in a bottom toolbar - the Claude-Code-style prompt.
        completer = SlashCompleter(_slash_help_map()).build()
        prompt_session = PromptSession(
            history=FileHistory(str(paths.data_dir() / "history")),
            completer=completer,
            complete_while_typing=True,
        )
    except ImportError:
        pass  # plain input() still works, just without history or editing

    def toolbar():
        c = state["config"]
        return f"  {c.active_profile} · {agent.provider.model} · mode:{c.security.mode}   /help"

    queued: list[str] = []  # prompts typed while the agent was working

    def _process(line: str) -> bool:
        """Handle one line (command or turn). Returns False to exit the REPL."""
        nonlocal cfg
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            keep = handle_command(line, state)
            cfg = state["config"]
            return keep
        speak = _speaker(state) if (state["voice_mode"] or cfg.speak_replies) else None
        run_turn(agent, ui, line, speak=speak)
        # Anything typed WHILE that turn ran was buffered by the terminal;
        # pull it in now so it runs next, in order - Claude-Code-style queueing.
        for ahead in _drain_typeahead():
            ui.info(f"↳ queued: {ahead}")
            queued.append(ahead)
        return True

    while True:
        # Run anything queued from type-ahead before prompting again.
        if queued:
            if not _process(queued.pop(0)):
                return 0
            continue

        if state["voice_mode"]:
            line = _voice_input(state)
            if line is None:
                continue
        else:
            try:
                line = (prompt_session.prompt("› ", bottom_toolbar=toolbar)
                        if prompt_session is not None else input("> "))
            except EOFError:
                ui.console.print()
                return 0
            except KeyboardInterrupt:
                ui.console.print("[dim](ctrl-d or /quit to exit)[/]")
                continue

        if not _process(line):
            return 0


def _drain_typeahead() -> list[str]:
    """Non-blocking read of any whole lines the user typed while the agent was
    busy (the terminal line-buffers them in cooked mode). Returns them in
    order; empty if none or not a TTY. This is what makes queueing work without
    a background thread fighting the prompt for stdin."""
    import select

    try:
        if not sys.stdin.isatty():
            return []
    except (ValueError, OSError):
        return []
    lines: list[str] = []
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                lines.append(line)
    except (OSError, ValueError):
        pass
    return lines


def _voice(state: dict):
    """Lazily build and cache the Voice engine on the agent context."""
    agent: Agent = state["agent"]
    voice = agent.ctx.state.get("voice")
    if voice is None:
        from .voice import Voice

        voice = Voice(state["config"])
        agent.ctx.state["voice"] = voice
    return voice


def _speaker(state: dict):
    ui: UI = state["ui"]

    def speak(text: str) -> None:
        try:
            _voice(state).speak(text)
        except Exception as exc:  # noqa: BLE001 - speech is a nicety, never fatal
            ui.warn(f"could not speak: {exc}")

    return speak


def _voice_input(state: dict) -> str | None:
    """Capture one spoken turn. Returns the transcript, "" to skip, or None if
    the user asked to leave voice mode (handled by the caller looping again)."""
    ui: UI = state["ui"]
    try:
        voice = _voice(state)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"voice unavailable: {exc}")
        state["voice_mode"] = False
        ui.info("dropped back to typing")
        return None

    ui.console.print("[dim](voice - speak after the prompt; Ctrl-C for a typed line)[/]")

    def on_state(phase: str) -> None:
        if phase == "waiting":
            ui.console.print("[cyan]listening...[/] ", end="")
        elif phase == "listening":
            ui.console.print("[green](hearing you)[/]")

    try:
        result = voice.listen(on_state=on_state)
    except KeyboardInterrupt:
        # Let the user drop to a typed line for one turn without leaving voice.
        ui.console.print()
        try:
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            state["voice_mode"] = False
            return None
    except Exception as exc:  # noqa: BLE001
        ui.error(f"voice capture failed: {exc}")
        state["voice_mode"] = False
        return None

    if result.aborted or not result.text:
        ui.info("(heard nothing)")
        return ""
    ui.console.print(f"[bold]you:[/] {result.text}")
    return result.text


def main(argv: list[str] | None = None) -> int:
    # Carry over config/data from the tool's former name before anything reads
    # it, so a rename never silently loses a user's setup.
    paths._migrate_from_old_name()
    args = build_parser().parse_args(argv)
    ui = UI(plain=args.plain, quiet=args.quiet)

    overrides = {
        "profile": args.profile,
        "model": args.model,
        "mode": "allow" if args.yolo else args.mode,
    }
    if args.no_stream:
        overrides["stream"] = False
    if args.tools:
        overrides["tools"] = [t.strip() for t in args.tools.split(",") if t.strip()]

    try:
        cfg = config_mod.load(args.config, overrides)
    except PAError as exc:
        ui.error(str(exc))
        return 2

    # Local models drown in a big tool list. If the user is on a local provider
    # and did not choose tools themselves, fall back to the curated local set.
    _tools_were_chosen = bool(args.tools) or _config_sets_tools(args.config)
    if cfg.profile.provider in ("ollama", "openai-compat") and not _tools_were_chosen:
        cfg.tools = list(cfg.local_tools)

    if args.doctor:
        return doctor(ui, cfg)

    # ---- service management (no agent needed) ------------------------------
    if args.install_service or args.uninstall_service:
        from . import service

        try:
            ui.info(service.install() if args.install_service else service.uninstall())
            return 0
        except PAError as exc:
            ui.error(str(exc))
            return 1

    # ---- daemon control (talk to an already-running daemon) ----------------
    if args.daemon_status:
        from . import client

        try:
            info = client.status()
        except PAError as exc:
            ui.error(str(exc))
            return 1
        ui.console.print(
            f"daemon pid {info.get('pid')}  up {info.get('uptime')}s  "
            f"turns {info.get('turns')}  {'busy' if info.get('busy') else 'idle'}\n"
            f"{info.get('profile')}:{info.get('model')}  "
            f"session {info.get('session')}  {info.get('tools')} tools"
        )
        return 0

    if args.daemon_stop:
        from . import client

        try:
            client.stop()
            ui.info("daemon stopping")
            return 0
        except PAError as exc:
            ui.error(str(exc))
            return 1

    # Route a prompt (or the whole REPL) to a running daemon rather than a
    # local cold-start agent.
    if args.daemon:
        from . import client

        speak = _remote_speaker(ui, cfg) if (args.voice or cfg.speak_replies) else None
        try:
            if args.prompt:
                ok = client.send(" ".join(args.prompt), ui=ui, speak=speak)
                return 0 if ok else 1
            return _daemon_repl(ui, cfg, speak=speak)
        except PAError as exc:
            ui.error(str(exc))
            return 1

    if args.warm:
        from .memory import Memory

        ui.info("warming the local embedding model (first run downloads ~80MB) ...")
        try:
            ui.console.print("  " + Memory(auto_install=cfg.auto_install_deps).warm())
            return 0
        except PAError as exc:
            ui.error(str(exc))
            return 1

    caps = capabilities.probe()

    try:
        provider = providers.build(cfg.profile, cfg)
    except PAError as exc:
        ui.error(str(exc))
        ui.info("run `arbin-assistant --doctor` to see what is configured")
        return 2

    gate = Gate(cfg.security)
    try:
        state = store_mod.build(cfg)
    except PAError as exc:
        ui.warn(f"state backend unavailable ({exc}); background tasks disabled")
        state = None
    # Routine startup notes (Wayland advice, which state backend) are available
    # on demand via /caps; showing them on every launch buries the welcome card.
    ctx = ToolContext(cfg, caps, gate, make_approver(ui, gate), store=state)
    try:
        registry = build_registry(cfg, caps)
    except PAError as exc:
        ui.error(str(exc))
        return 2

    session = None
    if args.session:
        candidate = paths.sessions_dir() / f"{args.session}.json"
        if not candidate.exists():
            ui.error(f"no session {args.session}")
            return 2
        session = Session.load(candidate)
    elif args.resume:
        session = Session.latest()
        if session is None:
            ui.info("no previous session - starting fresh")

    agent = Agent(cfg, provider, registry, ctx, caps, session)

    try:
        if args.serve:
            from .daemon import Daemon

            return Daemon(agent, ui=ui).serve()
        if args.prompt and args.auto:
            return 0 if run_autopilot(agent, ui, " ".join(args.prompt)) else 1
        if args.prompt:
            speak = _speaker({"agent": agent, "ui": ui, "config": cfg}) if (
                args.voice or cfg.speak_replies
            ) else None
            return 0 if run_turn(agent, ui, " ".join(args.prompt), speak=speak) else 1
        ui.banner(cfg.active_profile, provider.model, len(registry), agent.session.id)
        if cfg.security.mode == "allow":
            ui.warn("permission mode is 'allow' - tool calls will not ask first")
        if args.voice:
            ui.info("voice mode - speak your turns, or /voice to switch to typing")
        return repl(agent, ui, cfg, voice_mode=args.voice)
    finally:
        provider.close()
        agent.close()
        if browser := ctx.state.get("browser"):
            browser.close()
        if state is not None:
            state.close()


def _remote_speaker(ui: UI, cfg: config_mod.Config):
    """A speaker for daemon-client mode, where there is no local agent."""
    from .voice import Voice

    voice = Voice(cfg)

    def speak(text: str) -> None:
        try:
            voice.speak(text)
        except Exception as exc:  # noqa: BLE001
            ui.warn(f"could not speak: {exc}")

    return speak


def _daemon_repl(ui: UI, cfg: config_mod.Config, *, speak=None) -> int:
    """A REPL whose every turn goes to the running daemon over the socket."""
    from . import client

    ui.console.print(
        "[bold cyan]arbin-assistant[/] [dim](connected to daemon)[/]  "
        "[dim]/quit to disconnect[/]\n"
    )
    prompt_session = None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        paths.ensure_dirs()
        prompt_session = PromptSession(
            history=FileHistory(str(paths.data_dir() / "history"))
        )
    except ImportError:
        pass

    while True:
        try:
            line = prompt_session.prompt("> ") if prompt_session else input("> ")
        except EOFError:
            ui.console.print()
            return 0
        except KeyboardInterrupt:
            ui.console.print("[dim](ctrl-d to disconnect; the daemon keeps running)[/]")
            continue
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line == "/stop-daemon":
            client.stop()
            ui.info("daemon stopping")
            return 0
        try:
            client.send(line, ui=ui, speak=speak)
        except PAError as exc:
            ui.error(str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
