---
name: review-mr
description: "Review an EnTiSe branch, diff, or merge request for concrete regressions and missing verification. Use for technical review of changes to methods, core execution, or services."
---

# Review an EnTiSe change

## Establish the review scope

Identify the target and head commits. For a GitLab MR, use its target and
current diff. For a local branch targeting `develop`, start with
`git diff origin/develop...HEAD`. Check the worktree separately if the request
includes staged or unstaged changes, and identify those changes in the report.
Read [AGENTS.md](../../../AGENTS.md) and the scoped guidance for the files under
review.

## Check the changed behavior

- Trace the change through its normal caller. For a generator strategy, follow
  the [runner](../../../entise/core/runner.py) and
  [generator](../../../entise/core/generator.py). Check that declared inputs,
  upstream strategy results, and returned time series and summaries agree
  with what callers receive. A generator smoke test can use
  `raise_on_error=True` and assert the result for the expected object and type.
- Check units, column names, index alignment, and compatibility with existing
  inputs wherever the diff changes those contracts. For time-series joins,
  check the applicable label or positional alignment and look for shifted or
  missing values. If bundled reference data changes, check its source and the
  effect on consumers.
- If caching or shared state changes, check which inputs identify the result,
  when state is invalidated, and whether a cache hit matches an uncached run.
- Use the examples in the
  [method MR template](../../../.gitlab/merge_request_templates/Method_Merge_Request.md)
  for the invariants that apply. These include determinism across execution
  layouts, RC energy balance, and model bounds. For a bug fix, look for a test
  that fails with the base implementation and passes with the fix. Use local
  fixtures or mocked provider calls when running focused tests. Report which
  checks you ran and distinguish confirmed baseline failures from new ones.
- Check [Unreleased in the changelog](../../../CHANGELOG.md#unreleased) when
  generated outputs or the public API change.

## Report the review

Put actionable findings first. Give the file and line, the input or execution
path that triggers the issue, and the consequence. State the reviewed head,
test results, and any verification gaps. If there are no actionable findings,
say so. Recheck later changes before treating a new head as reviewed. Apply
fixes or publish comments only when the user requests those actions.
