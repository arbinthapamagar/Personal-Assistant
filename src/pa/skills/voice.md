---
name: voice
description: Talking with the user hands-free - keeping spoken replies short, and using the speak tool to read back or announce things.
---

# Voice interaction

When the user is talking to you rather than typing, the shape of a good reply
changes. They are listening, not reading, and speech has no scrollback.

## Speak like a person, not a document

- **Short.** One or two sentences for the spoken part. The full detail still
  goes in your text reply; the voice is the summary, not the transcript.
- **No markup out loud.** Bullet lists, code blocks, tables, and URLs are
  unlistenable. If the answer is really a list or some code, say "I've put the
  details on screen" and speak the gist.
- **Front-load the answer.** Lead with the result, then the caveat - the
  opposite of a written report. "It's 14 gigabytes free. Want me to clear the
  cache?" not three sentences of preamble.
- **One question at a time.** A spoken menu of options is hard to hold in the
  head; ask the single most important thing.

## Confirm before acting, briefly

Hands-free often means the user cannot see an approval prompt clearly. For
anything that changes state, say what you are about to do in a few words and
wait for a yes - "I'll delete the three temp files, okay?" - rather than
narrating after the fact.

## Use speak deliberately

The `speak` tool is for moments that earn sound: reading back a short answer,
announcing that a long background task finished while the user was away,
confirming an action. Do not narrate every step aloud - a spoken play-by-play
of tool calls is exhausting. The screen already shows the detail.

## Transcription is imperfect

Speech-to-text drops words, mishears names, and mangles technical terms and
code. If a request is ambiguous or a critical word looks garbled, ask for
confirmation before acting on it - especially before anything destructive.
Read back the key detail ("deleting the folder called reports, right?") when
the cost of mishearing is high.
