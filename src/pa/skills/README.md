# Skills

Each `.md` file here is a playbook the agent can load on demand. Only the
`description` from a skill's front matter sits in the system prompt; the body is
pulled in with the `use_skill` tool when the agent judges it relevant.

## Writing your own

Drop a Markdown file with front matter into one of:

- `~/.config/personal-assistant/skills/` — yours, on this machine
- `<project>/.pa/skills/` — travels with a repository

```markdown
---
name: deploy
description: How we ship this project - the exact steps, in order.
---

# Deploying

1. ...
```

Later paths win on name collision (project overrides user overrides bundled),
so a repo can specialise a general skill. Skills are read at launch.

Keep the description to one line that says *when* the skill applies — that line
is all the model sees when deciding whether to load the body.
