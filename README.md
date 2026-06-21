# Proctoring Tool — AI Build Kit

This folder is a complete prompt-engineering kit to build a lightweight, real-time Python proctoring tool using an AI coding agent (Claude Code, Cursor, etc.) one module at a time, without drift.

## Files
- **Agents.md** — the charter. Tech stack, rules, folder structure. The agent reads this every session.
- **coding_standards.md** — style, performance, and honesty-about-accuracy rules. Enforced on every module.
- **progress_tracker.md** — living status table. The agent updates this after every prompt.
- **prompts/01 → 11** — one self-contained build step each, in order.

## How to run this
1. Open a **fresh** chat/context with your AI coding agent.
2. Paste this instruction, then the contents of the next prompt file (start with `01_environment_setup.md`):
   > "Read Agents.md, coding_standards.md, and progress_tracker.md first, then execute this prompt only. Do not start the next module."
3. When it finishes, check it actually updated `progress_tracker.md`. If not, ask it to.
4. Start a **new fresh context** for the next prompt file. Don't chain multiple prompts in one long session — that's what causes drift/scope creep on long agent runs.
5. Repeat through prompt 11.

## Why this structure
- **Token efficiency**: each prompt is scoped to one file/module, so the agent never re-reads or re-reasons about unrelated code.
- **No drift**: `Agents.md`'s locked tech stack + "no new dependencies" rule stops the agent from wandering into heavier models that would kill performance.
- **Real accuracy, not a marketing number**: every detector ships with a confidence score and documented failure conditions; Prompt 11 produces a measured precision/recall report instead of an unverified "95-100%" claim.

## Tech Stack at a Glance
- MediaPipe Face Mesh + Pose → eye gaze, head pose, lip movement, shoulder tracking, face presence (all CPU-only, fast)
- YOLOv8n (Ultralytics) → person count, phone detection, secondary-device heuristic
- OpenCV → capture & rendering
- Plain JSON-lines event log + YAML config — no database, no cloud dependency, fully local and lightweight
