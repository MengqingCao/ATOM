# ATOMesh principles (project layer)

These sit on top of agent-team's default principles; where the two conflict, these win. None overrides a default today.

## One load path for every backend
A vLLM worker's load is sampled by the same LoadMonitor path as an ATOM worker's, and a failed read writes nothing, so routing falls back to the router's local counts rather than a stale value. Two paths drift, and a stale load is worse than none.
Source: [wuhuikx/LLM-Booster#709](https://github.com/wuhuikx/LLM-Booster/issues/709), [ROCm/ATOM#3566](https://github.com/ROCm/ATOM/pull/3566).

## Rust unit tests are tests when sizing a change
A `tests.rs` module or an inline `#[cfg(test)]` block counts as test lines against a task's estimate, not as code. Counting them as code flags a change as oversized when its code is inside the bound.
Source: Rulings on [#5](https://github.com/MengqingCao/ATOM/pull/5#issuecomment-7095196778) and [#6](https://github.com/MengqingCao/ATOM/pull/6#issuecomment-7095217709).

## Names match what the code does
Rename a variable, function or file when its behaviour changes, and after a fix sweep the codebase for the same pattern.
Source: ATOM `CLAUDE.md`, Critical Rules.
