---
name: self-development
description: Modifying your own source code safely - the checkpoint, edit, test, roll-back loop, and how to add a new tool or skill to yourself.
---

# Changing your own code

You can edit your own source. That is powerful and it is how you improve over
time - but a broken self-edit stops you from running at all, so it is done under
a safety loop, never freehand. Git is the seatbelt: every change is one command
from undone.

## The loop - never skip a step

1. **Locate.** `self_locate` to find your source tree and confirm it is a clean
   git checkout. If it reports you are running from an installed wheel, you
   cannot self-modify - say so and stop.
2. **Checkpoint.** `self_checkpoint` before touching anything. This commits the
   current working state so there is a known-good point to return to.
3. **Read before writing.** Read the file you intend to change in full. Match
   the surrounding style - you are extending a codebase with clear conventions
   (see the `coding` skill).
4. **Edit.** Make the smallest change that does the job, with `edit_file`.
5. **Test.** `self_test` immediately. If it reports FAILED, either fix it now
   with another small edit and re-test, or `self_rollback` and report what went
   wrong. Never leave yourself in a failing state.
6. **Confirm.** Tell the user what you changed, that tests pass, and that a
   **restart is required** for the new code to load (a running Python process
   does not pick up its own source edits live; if running under the daemon,
   `arbin-assistant --daemon-stop` then start it again, or the systemd service
   restarts it).

## Adding a new tool to yourself

Tools live in `src/pa/tools/`. To add one:

- Create or extend a module there with a `Tool` subclass: set `name`, `group`,
  `description`, a JSON-Schema `parameters`, and implement `run(self, args, ctx)`.
  Give it a `key()` for the permission gate and, if it touches a shared resource,
  a `lock_key()`.
- Add the class to that module's `TOOLS` list.
- If it is a new group, register the module name in `_LIGHT_GROUPS` (or
  `_HEAVY_GROUPS` for one with a heavy optional dependency) in
  `src/pa/tools/base.py`, and add the group to `tools.enabled`.
- Write a test in `tests/` and run `self_test`.

Look at an existing small tool (e.g. `src/pa/tools/web.py`) as a template before
writing a new one.

## Adding a skill

Skills are Markdown with front matter in `src/pa/skills/` (or, per-user, in
`~/.config/arbin-assistant/skills/`). Add a file with `name` and a one-line
`description`; the body is the playbook. No code change needed - skills are read
at launch.

## Rules that keep you safe

- **Checkpoint first, always.** An untested self-edit with no restore point is
  the one thing that can leave you unable to run.
- **One change at a time.** Change, test, confirm green, then the next thing.
  A batch of edits that fails is hard to bisect and easy to roll back too far.
- **Never edit while the tests are already red.** Get to green first.
- **Do not touch the safety or permission code to weaken it as a "fix."** If a
  permission check is in your way, that is the gate doing its job - tell the
  user, do not route around it in your own source.
- If a change needs a new dependency, add it to `pyproject.toml` and say so.
