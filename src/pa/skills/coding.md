---
name: coding
description: Writing or changing code in an existing project - the read-plan-change-verify loop, and how to match a codebase instead of imposing on it.
---

# Coding in someone else's codebase

The failure mode is not writing bad code. It is writing *plausible* code that
does not fit — wrong idiom, wrong layer, a helper that already existed two
files over. Reading is the cheap part; guessing is the expensive part.

## 1. Orient before typing

Do not open an editor on the first file whose name matches. Spend a few tool
calls establishing:

- **Where does this kind of thing live?** `code_search` for the concept
  ("permission check", "retry logic"), `search` for the exact symbol.
- **Does it already exist?** The most common wasted change is a second
  implementation of something already present under a different name.
- **What does the surrounding code look like?** Read a whole neighbouring file,
  not a fragment. You are matching its conventions, so you need to see them.
- **How is it tested?** The test file tells you the intended contract far more
  reliably than the implementation does.

Run these searches **in parallel** — they are independent.

## 2. Plan the change, then state it

Before editing, know: which files change, what each change is, and how you
will know it worked. For anything touching more than about three files, say
this out loud in two or three sentences first. A wrong plan is cheap to
correct; a wrong implementation is not.

## 3. Make the change fit

- **Match the local style** — naming, error handling, comment density, import
  order. The goal is a diff a maintainer would not be able to pick out as
  foreign.
- **`edit_file` over `write_file`** on anything that already exists. A whole-file
  rewrite hides the real change inside noise and loses anything you did not
  know was there.
- **Read the exact text before replacing it.** `edit_file` needs the old string
  verbatim, indentation and all.
- **Change one thing.** Do not reformat, rename, or tidy on the way past unless
  that was the request. Mention it instead.
- **No new dependency** without saying so and why.

## 4. Verify with something that can fail

An unverified change is a guess. In descending order of value:

1. Run the project's tests. Find the command from CI config, `Makefile`, or
   `package.json` — do not invent one.
2. Run the thing itself and observe the actual behaviour.
3. At minimum, confirm it imports/compiles.

Use `task_start` for a slow suite, keep working, and collect the result later.

**Report what the output said, not what you hoped.** If tests fail, say so and
paste the relevant lines. A confident "done" over a red suite destroys trust
permanently, and it is the single worst thing an agent can do.

## 5. Comments

Comment the *why*, never the *what*. If a line needs explaining, the reason it
exists is the interesting part — a constraint, a bug it works around, an
ordering that matters. Do not narrate the code, do not leave "changed this"
notes, and do not add a docstring that only restates the signature.
