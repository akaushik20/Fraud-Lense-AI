# Claude Code Instructions

## Code Style
- Write modular, reusable code — break logic into small, single-responsibility functions
- Keep functions short and focused; if a function does more than one thing, split it
- Use descriptive, self-explanatory names for variables, functions, and classes
- Avoid deeply nested logic; prefer early returns and guard clauses
- Do not over-engineer — solve the problem at hand without unnecessary abstraction

## Structure
- Group related logic into modules/files; avoid monolithic scripts
- Separate data loading, preprocessing, model training, and evaluation into distinct components
- Reuse existing utilities from `src/helper.py` before writing new ones

## Readability
- Prefer clarity over cleverness — code should read like plain English
- Keep lines concise; avoid cramming too much into a single expression
- Use consistent formatting and indentation throughout

## General
- Follow standard software engineering principles: DRY, KISS, YAGNI
- Write code that is easy to test and reason about
- Prefer explicit over implicit behavior
- Do not over generate
