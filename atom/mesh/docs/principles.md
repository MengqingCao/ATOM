# Mesh principles

These add to the default principles; nothing here overrides one.

## A PR stays reviewable

Each PR is at most 1000 lines and 15 files; work that is larger is split into stacked PRs. Reviewers on the upstream stack review one concern at a time. Source: owner's standing rule for the `ROCm/ATOM` 3569 stack.

## Build on the oldest supported toolchain

Mesh code builds on Rust 1.94.0 as well as the current toolchain, so an API new since then is not used. The CI image builds with 1.94. Source: the `fetch_update` decision on `ROCm/ATOM#3567`.

## Names say what the code does

A variable, function or file is renamed when its behaviour changes, and a fixed bug's pattern is swept across the codebase in the same change. Source: `CLAUDE.md`, Critical Rules.
