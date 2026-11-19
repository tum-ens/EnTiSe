# GitLab CI for EnTiSe

[The pipeline configuration](../../.gitlab-ci.yml) selects merge requests
targeting `develop` or `main` and pushes to those two branches.

## Verification jobs

- **Test** runs on Python 3.12 for `develop`. Merge requests targeting
  `main` and pushes to `main` test Python 3.11 through 3.14, with pandas 2.2.2
  on Python 3.11 and pandas 2.3 and 3 on newer versions. Both include the
  installation check and exclude tests marked `native_pylpg`.
- **Native PyLPG** runs the external engine tests on Python 3.12 and reports
  failures separately while the runner baseline is established. The local
  engine has failed with exit status -6. Tests that exercise validation without
  starting the engine stay in the main test job.
- **Lint and type checks** report Ruff lint, formatting, and mypy findings.
  This job is advisory during the rollout while existing findings are
  measured on the runner.

All verification jobs can start in parallel. Available runner slots
determine how many jobs GitLab runs at once.

## Verify the rollout

GitLab CI/CD must be enabled for the project. An eligible runner must accept
untagged jobs and run the declared Python container images. Check the
pipeline attached to an MR targeting `develop`, confirm the jobs execute,
and compare their results with the target branch. A pending pipeline or
an advisory failure is visible evidence to report in the MR.

The [contribution workflow](CONTRIBUTING.md#workflow) calls for comparing
the reviewed commit with the publication head. A required server-side
review check depends on a verified runner baseline.
