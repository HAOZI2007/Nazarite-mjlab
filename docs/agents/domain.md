# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the
codebase.

## Layout: multi-context

This repo is **multi-context**. Each subproject keeps its own `GLOSSARY.md` and its own
`docs/adr/`; a root `GLOSSARY-MAP.md` is the index that points at them, and the root `docs/adr/`
holds decisions that span contexts.

None of those files exist yet. They are created lazily by `/domain-modeling` (reached via
`/grill-with-docs` and `/improve-codebase-architecture`) the first time a term or a decision
actually gets resolved — the root `GLOSSARY-MAP.md` is written first, pointing at the per-context
`GLOSSARY.md` files, and each subproject's glossary is created when its first entry lands. Don't
create empty placeholders upfront.

## Contexts

| Context | Path | Vocabulary it owns |
| --- | --- | --- |
| Training framework | `Train/Nazarite/Nazarite-src/nazarite/` | `nazarite` package: env configs, MDP terms (rewards / terminations / curriculums / commands), RL runners, the WMP world model |
| Simulator | `Train/Nazarite/mjlab/` | `mjlab`: MuJoCo Warp stepping, sensors, terrain generator, viewer. Vendored upstream project with its own `AGENTS.md` and `CLAUDE.md` |
| Robot assets | `Train/Nazarite/MJCF-Manager/` | MJCF models and the tooling that patches them |

## Before exploring, read these

- **`GLOSSARY-MAP.md`** at the repo root if it exists, then the `GLOSSARY.md` of each context
  relevant to the topic (see the table above). If the map doesn't exist yet, fall back to reading a
  context's `GLOSSARY.md` directly when it happens to exist.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.
- **`<context>/docs/adr/`**: context-scoped decisions, e.g.
  `Train/Nazarite/mjlab/docs/adr/`.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest
creating them upfront.

## File structure

```
/
├── GLOSSARY-MAP.md                    ← index, written first when the first term is resolved
├── docs/
│   ├── adr/                           ← system-wide decisions
│   └── agents/                        ← this directory: issue tracker, triage labels, domain docs
└── Train/Nazarite/
    ├── Nazarite-src/
    │   ├── GLOSSARY.md                ← training-framework vocabulary
    │   └── docs/adr/                  ← context-specific decisions
    └── mjlab/
        ├── GLOSSARY.md                ← simulator vocabulary
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a
test name), use the term as defined in the relevant `GLOSSARY.md`. Don't drift to synonyms the
glossary explicitly avoids.

If the concept you need isn't in a glossary yet, that's a signal: either you're inventing language
the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
