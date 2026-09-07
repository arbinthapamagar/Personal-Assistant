---
name: debugging
description: Diagnosing a failure - reproduce, narrow, form one hypothesis at a time, and confirm the fix actually fixed it.
---

# Debugging

Discipline beats cleverness. The bug is almost never where it feels like it is,
and "I think it's probably X" applied three times in a row produces a mess with
the original bug still in it.

## Reproduce first

If you cannot make it fail on demand, you cannot know you fixed it. Get to a
single command that reproduces it. If you cannot reproduce it, say so — and
gather evidence (logs, versions, the exact input) rather than speculating in
prose.

## Read the whole error

The real cause is usually named in the part people skip: the innermost frame,
the "caused by", the first error rather than the last. Read the actual
traceback before theorising.

## Narrow before fixing

Cut the search space in half at a time. Does it fail with simpler input? At an
earlier commit (`git bisect`)? With one component stubbed out? Each answer
should eliminate possibilities, not just add detail.

Add temporary instrumentation freely — printing the actual value of the thing
you assume is fine is how assumptions die. Remove it before you finish.

## One hypothesis at a time

State it so it can be wrong: "the config is read before the env var is set, so
the override is lost." Then test *that*. Changing three things at once means a
green result teaches you nothing about which mattered.

## Fix the cause

A fix that makes the symptom disappear without explaining it is not a fix. If
you genuinely cannot find the cause and are applying a mitigation, say that
plainly — a labelled workaround is fine, a workaround presented as a fix is
not.

## Confirm

Re-run the reproduction. Then run the broader suite, because the interesting
question is what else your change touched. If a test now fails that passed
before, that is your problem and not a coincidence.

Finally: consider whether a regression test belongs here. A bug that shipped
once can ship twice.
