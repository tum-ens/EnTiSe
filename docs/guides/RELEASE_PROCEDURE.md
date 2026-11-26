# Release procedure

EnTiSe uses annotated Git tags and Hatch VCS for package versions. Stable
releases use `vMAJOR.MINOR.PATCH` and release candidates use
`vMAJOR.MINOR.PATCHrcN`, for example `v2.0.0rc1`. A release tag is immutable.
If the source changes or a published version is unusable, cut a new
candidate or patch release from a reviewed commit.

## Configure LRZ GitLab and the package indexes

A maintainer configures these settings before the first rehearsal.

- Protect `main`. Disable direct pushes and force pushes, restrict merges to
  maintainers, and require review and a successful pipeline for merges. The
  tag pipeline checks the tagged commit against the first-parent history of
  `main`. The branch rules ensure those commits have passed review.
- Protect tags matching `v*` and limit tag creation to maintainers. Check
  inherited or overlapping tag rules for broader access.
- In **Settings → CI/CD → Protected environments**, protect `testpypi` and
  `pypi`. Allow only the release maintainers to deploy, and add them as
  individual users. The Maintainers role also includes everyone who
  inherits the Owner role from the parent groups.
- LRZ GitLab doesn't offer deployment approvals; this was checked for 2.0.0.
  So `pypi` is guarded by its deploy access, the manual
  `release:upload:pypi` job and the job's confirmation prompt: only a named
  release maintainer can start the upload. If the tier gains approvals, also
  set a required deployment approval on `pypi` from the release maintainers,
  and keep pipeline-triggerer self-approval disabled.
  [GitLab deployment approvals](https://docs.gitlab.com/ci/environments/deployment_approvals/)
- Create separate API tokens scoped to the `entise` project on TestPyPI and
  PyPI. In **Settings → CI/CD → Variables**, add `TEST_PYPI_API_TOKEN` with
  environment scope `testpypi` and `PYPI_API_TOKEN` with scope `pypi`.
  Mark both variables protected, masked and hidden, and disable variable
  expansion. Use the exact environment names and restrict pipeline variable
  overrides and access to the publishing runner. PyPI Trusted Publishing
  supports GitLab.com. The self-managed LRZ project uses API tokens.
  [GitLab CI/CD variables](https://docs.gitlab.com/ci/variables/),
  [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- Confirm a trusted runner can execute jobs on the protected tags. Verify
  that the two upload jobs receive only their environment's respective token.
  Build, verification and post-upload jobs use public read access to the
  indexes.

## Prepare and review the source

1. Start a release issue and use the **Release_Checklist** merge request
   template for the release into `main`. Summarize the release in
   [CHANGELOG.md](../../CHANGELOG.md), move the entries out of `[Unreleased]`
   into a dated release heading, and review the migration notes and examples.
   Date the heading with the day you plan to push the final tag. If the tag
   lands on a later day, correct the date on `develop` afterwards; don't cut
   a new candidate for it.
   `README.rst` is the package's PyPI project page, so make it final before
   the candidate tag. Every tag publishes the README it contains.
2. Merge the release only after review and the required checks pass,
   including strict Sphinx and the supported Python and pandas matrix.
   Native PyLPG is the only advisory job; review its result before tagging.
3. Fetch `main` and its tags, confirm the exact reviewed merge commit and
   record its SHA in the release issue. Use the same GitLab project for the
   review, tag and release pipeline.

## Rehearse on TestPyPI

Tag the reviewed commit with an unused candidate version and push just that
tag. Substitute the reviewed merge SHA in the example. Create every release
tag annotated (`git tag -a`). Hatch VCS takes the version from `git describe`,
which prefers annotated tags and, of two on one commit, the newer one. So a
final tag on a candidate's commit must be annotated and created after it.

```bash
git fetch origin refs/heads/main:refs/remotes/origin/main --tags
reviewed_sha=REPLACE_WITH_REVIEWED_MERGE_SHA
tag=v2.0.0rc1
git log -1 "$reviewed_sha"
git tag -a "$tag" "$reviewed_sha" -m "Release candidate $tag"
git push origin "refs/tags/$tag"
```

1. Wait for the tag pipeline's build and verification stages. The build
   creates one sdist and one wheel using the tag and saves them with
   `dist/SHA256SUMS`. The required release jobs run the support matrix,
   strict Sphinx and clean installs of the wheel, the wheel with
   `pv-pvlib`, and the sdist. Review the results and the advisory logs.
2. Download the `release:build` artifacts and verify both files from the
   artifact directory with `sha256sum --check --strict SHA256SUMS`. Save the
   manifest with the release issue. GitLab retains these artifacts for 30 days
   by default. Keep the build and verification artifacts in GitLab if the
   approval window may be longer.
3. Have a release maintainer start `release:upload:testpypi`. Review
   `release:verify:testpypi`, which uses the exact version's TestPyPI JSON
   record, verifies the published and downloaded hashes for both files, and
   installs the downloaded wheel in a fresh environment. Dependencies come
   from PyPI. Check the rendered project description and record the pipeline
   link and result. Candidate tags have no production publishing job.
4. Install the candidate yourself as well, in a clean environment, with the
   extras users install, and run an example. Take only the wheel from
   TestPyPI and every dependency from PyPI. Never mix the two indexes:
   TestPyPI may hold an unrelated package under a dependency's name.
   Compare the wheel's SHA-256 with the manifest. Right after an upload,
   TestPyPI's index can lag behind its JSON API; `--no-cache-dir` keeps pip
   from reusing an index page it fetched before the version appeared.

   ```bash
   version=2.0.0rc1
   work=$(mktemp -d)
   (
     cd "$work"
     uv venv --seed --python 3.12 rehearsal-env
     rehearsal-env/bin/python -m pip download --no-cache-dir --no-deps --only-binary :all: \
       --index-url https://test.pypi.org/simple/ -d wheel "entise==$version"
     wheel=$(ls wheel/entise-*.whl)
     sha256sum "$wheel"
     uv pip install --python rehearsal-env/bin/python --default-index https://pypi.org/simple/ \
       "entise[hvac-1r1c,pv-pvlib] @ file://$PWD/$wheel"
     rehearsal-env/bin/python -I -c 'from importlib.metadata import version; import entise, sys; assert entise.__version__ == version("entise") == sys.argv[1]' "$version"
   )
   ```
5. If the rehearsal identifies a source issue, fix it through review and
   repeat the rehearsal with a new candidate tag.

## Publish the final release

1. After a successful rehearsal, create and push the unused final tag
   (for example, `v2.0.0`) on the reviewed commit, annotated and in the same
   way as the candidate. Follow the same build,
   checksum and TestPyPI steps for the final tag. TestPyPI now verifies the
   final version's files, which are the files promoted to PyPI.
2. Review the final tag, commit, manifest and successful
   `release:verify:testpypi` result. A release maintainer with deploy access
   to `pypi` then starts `release:upload:pypi` and confirms the prompt,
   after approving the deployment where the tier has approvals. The job
   compares the build manifest with the TestPyPI verification record and
   uploads the matching wheel and sdist.
3. Wait for `release:verify:pypi` to verify both downloads and install the
   exact version in a clean environment. Record that result and review the
   release on [PyPI](https://pypi.org/project/entise/). For a second clean
   install from PyPI, use the version you published.

   ```bash
   version=2.0.0
   work=$(mktemp -d)
   (
     cd "$work"
     uv venv --python 3.12 verify-env
     uv pip install --python verify-env/bin/python --default-index https://pypi.org/simple/ "entise==$version"
     verify-env/bin/python -I -c 'from importlib.metadata import version; import entise, sys; assert entise.__version__ == version("entise") == sys.argv[1]' "$version"
   )
   ```
4. Create the GitLab Release for the final tag with the changelog notes,
   pipeline link and SHA-256 manifest.
5. Merge `main` back into `develop` through a merge request from `main`,
   with a merge commit. Don't squash, and don't delete the source branch.
   `develop` then contains the release tag, so its development versions
   count from it. Confirm that `git describe origin/develop` starts with the
   new tag, for example `v2.0.0-2-g…`.
6. Check that Read the Docs built the tag and that the stable documentation
   shows the new version.

If an upload fails partway, retry the upload job with the retained artifacts.
The publisher can reuse already uploaded files with matching hashes. A digest
mismatch or expired artifact requires investigation before proceeding. Leave
existing tags and published files intact. If users need to avoid a bad
production version, yank it on PyPI and release a reviewed patch.
