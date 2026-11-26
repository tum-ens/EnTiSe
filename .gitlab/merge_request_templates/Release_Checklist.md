## Release checklist

### Before merging to main

- [ ] Record the intended version and release issue.
- [ ] Review the changelog, migration notes, affected examples and the README,
      which is the PyPI project page.
- [ ] Review the support matrix, strict Sphinx and advisory job results.
- [ ] Confirm the protected main branch, release tags, environments with
      deploy access for the named release maintainers only, runner access
      and separately scoped tokens.

### After merging to main

- [ ] Record the reviewed commit SHA and push the immutable, annotated candidate tag.
- [ ] Verify the candidate wheel and sdist SHA-256 manifest, clean installs
      and exact-version TestPyPI download and install.
- [ ] Record the candidate pipeline result and check the package page.
- [ ] Push the immutable final tag on the reviewed commit.
- [ ] Verify the final tag's pipeline, retained checksums and TestPyPI upload.
- [ ] Start the protected PyPI publishing job as a named release
      maintainer and record who started it.
- [ ] Verify the published checksums and exact-version PyPI install.
- [ ] Create the GitLab Release with the notes, pipeline link and manifest.
- [ ] Merge main back into develop with a merge commit and check that
      `git describe origin/develop` starts with the release tag.
- [ ] Check that Read the Docs shows the release as stable.
