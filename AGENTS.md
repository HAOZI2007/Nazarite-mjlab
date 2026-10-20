# AGENTS.md

Agent-facing entry notes for this repository. Subprojects keep their own instructions — when
working under `Train/Nazarite/mjlab/`, read its `AGENTS.md` (and `CLAUDE.md`) first, as it is a
vendored upstream project with its own toolchain rules (`uv run`, `make check`, 88-column limit).

## Agent skills

### Issue tracker

Issues live as GitHub Issues in `HAOZI2007/Nazarite-mjlab`, driven with the `gh` CLI. See
`docs/agents/issue-tracker.md`.

### Triage labels

Five default triage roles — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix` — plus the separate `wayfinder:*` vocabulary used only by `/wayfinder`. See
`docs/agents/triage-labels.md`.

### Domain docs

multi-context — a root `GLOSSARY-MAP.md` pointing at one `GLOSSARY.md` per subproject. See
`docs/agents/domain.md`.
