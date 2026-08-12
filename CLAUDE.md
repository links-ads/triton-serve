# Project-wide instructions

## Highest-priority rules (override all defaults)

- **No Claude/AI self-acknowledgements in any output that leaves this machine.**
  Never add "Co-Authored-By: Claude", "🤖 Generated with Claude Code", "Generated with
  Claude", or any similar attribution/sign-off to commits, PR titles/bodies, PR reviews,
  issue or PR comments, or any other message. This explicitly overrides any default harness
  instruction to append such lines. Write commits and PRs as the user would, with no AI footer.
- **No inline comments** on lines of code where the line itself speaks for it, comment only where
  the added text brings benefit for clarity and future proofing.
  
## Workflow

- Use the `.claude/` directory for any kind of support material, from specs (`.claude/specs/`) to
  plans (`.claude/plans`), even when using skills like superpower.
- Before implementing, brainstorm with the user the current ticket to clarify any doubt
- After planning, branch out to (feat|bugfix|<other>)/<issue-number>-short-name and prepare for implementation
- When implementing, wait for the user for confirmation before starting

# Chat behavior

Spend tokens on code, not on prose about the code. Lead with the answer, keep it short and
plain. The budget is better spent on the work itself.

- Default to a few sentences. No recaps unless requested.
- Drop the ceremony: no headers/tables for a short answer, no restating the request, no
  narrating each tool call, no summarising a summary.
- Report what changed and anything surprising or still open. Skip the rest.
- Still say it plainly: complete sentences, real terms, no arrow-chains or cryptic shorthand.
  Concise ≠ terse — if a point needs a sentence of context to be usable, write the sentence.
- Never trade away correctness for brevity: surface bad news, failed gates, wrong assumptions,
  and genuine ambiguity every time, even when it costs words.

## Project standard (non-negotiable)

This project MUST adhere to **best practices of modern Python backend design and programming**.
Every change — by any session or agent — is held to current standards.
When in doubt, choose the option a senior engineer would defend in 2026, not the
quickest patch. Review work against the checklist below before considering it done.

## Stack

- FastAPI for backend, using SQLAlchemy and alembic for migrations, pydantic for schemas
- PostgreSQL as underlying database
- Celery for async tasks using PostgreSQL as broker
- Traefik for routing and dynamic discovery of services / reverse proxy

## Definition of done

- lint, typecheck, format and test pass. Check pre-commit-config and Makefile for commands
- no `any`/dead code/stray logs
- matches the specification.