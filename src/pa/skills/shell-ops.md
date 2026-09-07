---
name: shell-ops
description: System administration and shell work - inspecting the machine, managing services and packages, and being careful with destructive or privileged commands.
---

# Shell and system operations

You are running as the real user on a real machine. Commands have real
consequences and there is no undo on most of them.

## Look before you act

- **Inspect before you change.** `cat` the config before editing it, `ls` the
  directory before `rm`, `git status` before `git checkout`. State-changing
  commands deserve a read first.
- **Dry-run when offered.** `rsync --dry-run`, `git ... --dry-run`,
  `apt --simulate`. Show the user what *would* happen for anything sweeping.
- **Prefer specific over broad.** `rm ./build/*.o`, never `rm -rf` with a
  variable in the path you have not printed.

## Destructive and privileged commands

Some patterns are blocked outright and cannot run — recursive deletes of home
or root, formatting, raw block-device writes, `curl | sh`, and so on. That is a
floor, not the whole of good judgement.

Above that floor, anything irreversible or system-wide gets a sentence of
confirmation first, even when policy would allow it: deleting data, `sudo`
changes to system files, dropping a database, killing processes you did not
start, changing firewall or network config.

`sudo` specifically: name what you are about to run and why before you run it.
Never store a password in a command, a file, or a memory. If a command needs a
password interactively, ask the user to run it themselves — suggest the
`! <command>` prefix in their prompt.

## Diagnosing the machine

Reach for the right tool: `ss -tlnp` for listening sockets, `journalctl -u
<service>` for a service's logs, `systemctl status`, `df -h` and `du -sh *` for
disk, `free -h` and `ps aux --sort=-%mem` for memory, `ip a` for interfaces.
Run independent checks in parallel.

## Long-running commands

Background anything slow with `task_start` — package upgrades, large downloads,
backups, builds — then keep working and check back. Do not sit and wait on a
foreground command that will take minutes.

## Report what happened

Paste the exit status and the relevant output. "Done" without evidence is not
enough for anything that mattered, and "it worked" over a non-zero exit is
worse than saying nothing.
