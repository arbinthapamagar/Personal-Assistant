---
name: parallel-work
description: Using concurrent tool calls, background tasks, and sub-agents so slow work overlaps instead of queueing.
---

# Working in parallel

This agent runs tool calls concurrently and can push long jobs into the
background. Most agents waste enormous wall-clock time by doing independent
things one at a time. Do not.

## Batch independent calls

Issue them in one turn. They execute at once.

Good candidates: reading several files, several searches with different
phrasings, `git status` + `git log` + `git diff`, checking several services.

The rule is dependency, not similarity: if call B needs A's *output*, they must
be sequential. If it does not, they go together.

Some tools serialize themselves regardless, because the underlying resource is
singular — browser tabs, keyboard and mouse, two writes to one file. You do not
have to reason about this; it is handled. Batch anyway.

## Background anything slow

`task_start` returns immediately and the job keeps running — across the end of
the conversation, if need be.

Use it for: test suites, builds, installs, downloads, migrations, long scripts.

The pattern that pays off:

1. `task_start` the slow thing **first**, before anything else.
2. Do the useful work that does not depend on it — read code, plan, search.
3. `task_output` or `task_wait` when you actually need the result.

Starting a five-minute build and then sitting on `task_wait` for five minutes
is the same as not having backgrounded it. Fill the gap.

## Delegate wide work to sub-agents

`delegate` runs sub-agents in parallel, each with its own context, each
returning a summary.

It earns its cost when a task fans out — "review these six files", "check each
of these services", "investigate these four possible causes" — and especially
when the reading would otherwise flood this conversation's context with
material only needed once.

It is the wrong tool for one focused change: a sub-agent cannot ask you a
question, and it does not share your context. Give each one a self-contained
brief and say exactly what to report back.

## Know when not to

Parallelism has a cost: interleaved output is harder to follow, and a batch of
eight prompts is worse than one. For two quick sequential calls, just do them.
Reach for this when the work is genuinely wide or genuinely slow.
