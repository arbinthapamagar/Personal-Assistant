# Personal Assistant

An AI agent that runs on your own machine and is driven from your terminal. It
can run shell commands, read and edit files, drive Chrome, and control your
keyboard and mouse. The model behind it is swappable at runtime — Claude,
OpenAI, Gemini, a local Ollama model, or any OpenAI-compatible endpoint.

```
$ pa
personal assistant  claude:claude-opus-5  19 tools  session 4f2a9c1b3e77
/help for commands, /quit to exit

> which of my projects have uncommitted changes?
-> shell  command='for d in ~/Arbeen/Development/*/; do ...'
   PersonalAssistant  3 modified
   invoice-tool       1 untracked
Two of them: PersonalAssistant has 3 modified files, invoice-tool has 1 untracked.
```

## What makes it more than a chat loop

- **Parallel tools** — when the model asks for several tools at once, they run
  concurrently (measured ~6x on independent work). Resources that must not
  overlap — one browser, one keyboard, one file — serialize themselves.
- **Background tasks** — `task_start` launches a slow job (build, test suite,
  download) and returns immediately; it keeps running even if `pa` exits, and
  any later session can check on it.
- **Sub-agents** — `delegate` fans work out to several short-lived agents at
  once, each read-only, each returning a summary. For wide work that would
  otherwise bury the main conversation.
- **Memory** — `memory_save` / `memory_search` carry knowledge between
  sessions; `code_index` / `code_search` find code by meaning. Both run on a
  local vector database with CPU embeddings — nothing leaves the machine.
- **Skills** — playbooks (`coding`, `debugging`, `research`, `parallel-work`,
  `security-testing`, `shell-ops`, `voice`) the agent loads on demand, and you
  can add your own.
- **Voice** — talk to it and hear it back, entirely locally: `pa --voice`.
  Speech-to-text is faster-whisper, text-to-speech is Piper, and capture runs
  through the audio tools already on the machine. No cloud, no API key.
- **Daemon** — one warm agent (models, memory, browser, tasks all live) reachable
  from any terminal, script, or keyboard shortcut over a private socket. Install
  it as a systemd user service and it is always there: `pa --install-service`.

## Install



On any machine with Python 3.10+:

```bash
git clone <your-repo-url> personal-assistant
cd personal-assistant
./install.sh
```

Or without cloning first:

```bash
curl -fsSL https://raw.githubusercontent.com/<you>/<repo>/main/install.sh | \
  PA_REPO_URL=https://github.com/<you>/<repo>.git bash
```

The installer creates its own virtualenv under `~/.local/share/personal-assistant`,
links `pa` into `~/.local/bin`, and prints what your machine can and cannot do.
It never installs into system Python, so the "externally managed environment"
error on Ubuntu, Debian, and Fedora cannot happen.

Then give it a model — any **one** of these is enough:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # pa
export OPENAI_API_KEY=sk-...          # pa -p gpt
export GEMINI_API_KEY=...             # pa -p gemini
ollama serve                          # pa -p local   (no key, fully offline)
```

Heavy optional packages (provider SDKs, Playwright) install themselves the
first time you use a feature that needs them. To get everything up front:
`PA_EXTRAS=all ./install.sh`.

## Switching models

Profiles are named provider+model combinations. Switch at launch or mid-conversation;
conversation history carries across the switch.

```bash
pa -p local                    # start on a local model
pa -p gpt -m gpt-5-mini        # override the model too
```

```
> /profile              # list them
> /profile local        # switch now — history is kept
> /model llama3.3:70b   # change just the model
> /models               # what this provider offers
```

Supported `provider` values: `anthropic`, `openai`, `google`, `ollama`, and
`openai-compat` for anything else speaking the OpenAI wire format (Groq,
Together, OpenRouter, vLLM, LM Studio, llama.cpp). Only `anthropic` and
`openai` pull an SDK; the other two talk HTTP directly.

## Always-on daemon

Cold-starting a fresh agent for every question throws away the warm model
client, the loaded memory, and any open browser. The daemon keeps one running
and lets you reach it from anywhere:

```bash
pa --serve                     # run it in the foreground (Ctrl-C to stop)
pa --install-service           # or: run it as a systemd user service, forever
```

Then, from any terminal, script, or a keyboard shortcut:

```bash
pa --daemon "what changed in my repo today?"   # one-shot to the running daemon
pa --daemon                                     # a REPL wired to the daemon
pa --daemon-status                              # pid, uptime, model, busy/idle
pa --daemon-stop
```

Because the agent is warm, replies come back without startup cost and with all
state intact — the same conversation, the same indexed memory, the same browser
tab. Bind `pa --daemon "..."` to a global hotkey and you have an assistant one
keypress away anywhere on the desktop.

**How it stays safe.** The socket is a Unix socket in `$XDG_RUNTIME_DIR` at mode
`0600` — private to your user, never a network port. When a tool needs approval,
the daemon forwards the question to the terminal that submitted the request and
waits for your answer there; a request submitted non-interactively (a scheduled
job) cannot approve anything the policy does not already allow. Turns run one at
a time; `--daemon-status` and `--daemon-stop` answer even mid-turn.

The systemd service is a **user** service (never system/root): it runs as you,
with your sessions and permissions, starts at login, and — since your account
has linger enabled — keeps running after you log out. Manage it the usual way:

```bash
systemctl --user status personal-assistant
journalctl --user -u personal-assistant -f
pa --uninstall-service
```

## Permissions

Every tool call is reduced to a key — `shell:git push`,
`files.write:/etc/hosts` — and checked before it runs. By default you are asked:

```
╭─ approval needed ───────────────────────────╮
│ write /etc/hosts - OUTSIDE the workspace    │
│                                             │
│ key: files.write-outside-workspace:/etc/hosts│
╰─────────────────────────────────────────────╯
  y run once   a always allow files.write-outside-workspace:*   n skip   d always deny
```

Answer `a` and that pattern stops asking for the rest of the session. To make it
permanent, put it in the config:

```yaml
security:
  mode: ask                 # ask | allow | deny
  allow:
    - "files.read:*"
    - "shell:git status*"
  deny:
    - "shell:*docker system prune*"
```

`mode: allow` (or `PA_YOLO=1`) stops all prompting. Above that, a **denial
floor** still applies — `rm -rf /`, `mkfs`, raw writes to block devices,
`curl … | sh`, reading `/etc/shadow`, private SSH and cloud credentials, force
pushes. Each floor rule has an id, and you can disable specific ones
deliberately:

```yaml
security:
  unrestrict:
    - force-push        # now allowed
    # every other floor rule still holds
```

`/caps` warns when any floor rule is disabled. The rules and their ids are in
`src/pa/security.py` (`HARD_DENY_RULES`); read them before trusting this with
your machine.

## What it can do on your machine

Run `pa --doctor` for a per-machine report. Briefly:

| Group | Tools | Notes |
|---|---|---|
| `shell` | `shell`, `cwd` | Working directory persists between calls. |
| `files` | `read_file`, `write_file`, `edit_file`, `list_dir`, `search` | Uses ripgrep when installed. |
| `web` | `web_fetch`, `web_search` | Search has no-key (DuckDuckGo), SearxNG, and Brave backends. |
| `tasks` | `task_start`, `task_list`, `task_output`, `task_wait`, `task_cancel` | Background jobs that outlive the session. |
| `memory` | `memory_save`, `memory_search`, `memory_forget`, `code_index`, `code_search` | Local Chroma vector DB, CPU embeddings. |
| `skills` | `use_skill` | Load a playbook on demand. |
| `agents` | `delegate` | Parallel read-only sub-agents. |
| `voice` | `speak` | Say something aloud; the agent can choose to. |
| `browser` | `browser_open`, `browser_read`, `browser_click`, `browser_type`, `browser_tabs`, `browser_screenshot` | Real Chrome over the DevTools protocol. |
| `desktop` | `desktop_type`, `desktop_key`, `desktop_click`, `desktop_windows`, `desktop_focus`, `desktop_screenshot` | Backend depends on your session. |

### State backend

Background tasks and cross-session coordination need shared state. `pa` uses
**SQLite** by default (zero setup, WAL mode so several `pa` processes coexist).
If `redis` is reachable it uses that instead, which upgrades task-completion
notifications from polling to instant pub/sub. `state_backend: auto` (the
default) prefers redis and silently falls back to SQLite, so a machine that
loses redis keeps working.

### Memory and embeddings

The `memory` group runs a local [Chroma](https://www.trychroma.com/) database
with the bundled `all-MiniLM-L6-v2` ONNX model (384-dim, CPU). The model is
~80MB, fetched once to `~/.cache/chroma`. Run `pa --warm` after install to
download it deliberately rather than mid-conversation. **Nothing is sent to any
API for indexing or search** — this is why it works offline and costs nothing
to run.

This is retrieval, not fine-tuning: no model weights change. The agent gets
better over time because it recalls past decisions and loads relevant
playbooks, not because it is retrained.

### Browser

The agent attaches to a real Chrome with `--remote-debugging-port`, so it sees
pages as you would. **By default it drives a dedicated Chrome profile, not your
logged-in one** — attaching to your daily profile would hand the agent every
live session you have. Opt in deliberately:

```yaml
profiles:
  claude:
    provider: anthropic
    extra:
      browser:
        use_main_profile: true    # agent inherits YOUR logins
```

With `use_main_profile`, Chrome must already be running with the debug port, or
be fully quit so the agent can start it — a running Chrome ignores the flag.

### Voice

Talk to it, hands-free:

```bash
pa --voice          # spoken input, spoken replies
```

In the REPL, `/voice` toggles it and `/say <text>` speaks a line. Everything is
local:

- **Speech in** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
  a CPU build of Whisper. The `base` model (~140MB) downloads on first use.
  A spoken turn ends automatically after a short pause — press-to-talk, speak,
  stop.
- **Speech out** — [Piper](https://github.com/rhasspy/piper) neural voices when
  a voice model is present, falling back to `espeak-ng` (instant, robotic) when
  not. Drop a `<name>.onnx` (+ `.onnx.json`) into
  `~/.local/share/personal-assistant/voices/`, or set `voice.voice_model`.
  Grab voices from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).
- **Capture and playback** — `arecord`/`parecord` and `aplay`/`paplay`, already
  present on any PipeWire or ALSA system. No PortAudio to compile.

Set `speak_replies: true` to have replies read aloud even in text mode. The
`speak` tool lets the agent choose to talk — e.g. announcing that a background
task finished while you were away.

Nothing here calls a cloud API, so voice works offline and costs nothing to run.

### Desktop

Backend is a platform fact, not a preference:

- **X11** — `xdotool`. Can target a specific window.
- **Wayland** — `ydotool`, injecting at the kernel level via `/dev/uinput`.
  It has **no concept of windows**: text goes to whatever has focus. Focus
  first, then type. Window listing only sees XWayland apps. This is Wayland
  working as designed, not a bug.
- **Windows / macOS** — `pyautogui`.

Wayland setup, once:

```bash
sudo apt install ydotool
sudo systemctl enable --now ydotoold
```

Screenshots are separate, because the compositor owns the screen. On GNOME
Wayland, `gnome-screenshot` is used and `grim` does not work. X11 tools
(`scrot`, `maim`, `import`) are **never** selected on Wayland: they appear to
succeed and write a fully black image, so the agent would confidently describe
an empty screen. Captures are checked for an all-black frame and rejected.

## Configuration

```bash
pa            # then: /config
```

Creates `~/.config/personal-assistant/config.yaml`. See `config.example.yaml`
for every option with comments. Precedence, lowest to highest: built-in
defaults → config file → environment (`PA_PROFILE`, `PA_MODEL`, `PA_YOLO`) →
command-line flags. A machine with no config file at all still runs.

## Sessions

Conversations are saved to `~/.local/share/personal-assistant/sessions`.

```bash
pa --resume              # continue the newest
pa --session 4f2a9c1b    # continue a specific one
```

```
> /sessions   /stats   /new
```

## Commands

```
/help  /quit                    /profile [name]   /model [name]   /models
/tools  /caps  /check           /mode [ask|allow|deny]
/audit  /stats                  /new  /sessions
/voice  /say <text>             /config  /cwd [path]
```

## One-shot and scripting

```bash
pa "summarise today's git log"
pa -q "how much free disk?"                 # answer only, no tool chatter
pa --mode allow -q "run the test suite"     # unattended
```

## Prompt injection

Content from web pages and files reaches the model labelled as untrusted data,
and the system prompt tells it that text inside such content is never an
instruction. That is a mitigation, not a guarantee. Treat an agent that can
both read the web and run shell commands as a real attack surface: keep
`mode: ask` for anything destructive, and prefer the dedicated Chrome profile.

## Layout

```
src/pa/
  cli.py            argument parsing, REPL, slash commands
  agent.py          the provider-neutral agent loop
  messages.py       canonical Message / ToolCall / Completion types
  providers/        one adapter per backend; base.py is the contract
  tools/            one module per group; base.py is the Tool contract
  security.py       the permission gate and hard denials
  capabilities.py   machine probe (session type, backends, binaries)
  config.py         layered config and profiles
  session.py        conversation persistence
  prompts.py        system prompt assembly
  deps.py           lazy, on-demand dependency installation
  daemon.py         persistent agent over a Unix socket
  client.py         talks to a running daemon
  service.py        systemd user-service install/uninstall
  concurrency.py    parallel tool scheduler
  store.py          shared state (sqlite / redis)
  tasks.py          background jobs
  memory.py         semantic memory + code search (Chroma)
  voice.py          local speech in and out
  skills.py         on-demand playbooks
```

Adding a provider is one file implementing `Provider.complete()` plus a line in
`providers/registry.py`. Adding a tool is a `Tool` subclass with a JSON Schema
plus a line in its module's `TOOLS` list. Neither requires touching the agent
loop.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite runs with no API key and no network: a scripted fake provider drives
the real loop, real tools, and the real permission gate.

## Known limits

- The agent loop is synchronous — one tool call at a time, even when the model
  requests several.
- No MCP server support yet.
- Windows and macOS code paths are written but untested; Linux/X11/Wayland are
  verified.
- `openai-compat` endpoints vary in tool-calling quality. Small local models
  often call tools with malformed arguments.
