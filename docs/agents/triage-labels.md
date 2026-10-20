# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual
label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label
string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

## Also in play

`/wayfinder` uses its own vocabulary — `wayfinder:map`, `wayfinder:research`, `wayfinder:prototype`,
`wayfinder:grilling`, `wayfinder:task`. Those are not triage roles and are documented in
[`issue-tracker.md`](./issue-tracker.md#labels-in-this-repo). Don't apply them during triage.

The repository's default GitHub labels (`bug`, `enhancement`, `documentation`, `question`,
`good first issue`, `help wanted`, `invalid`, `duplicate`) are left as-is. They are optional
descriptors, not part of the triage state machine — a triaged issue always carries exactly one of
the five role labels above.
