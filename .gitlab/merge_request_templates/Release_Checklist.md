## Release checklist

### Before merging to main

- [ ] Record the intended version and release issue.
- [ ] Review the changelog, migration notes and affected examples.
- [ ] Review the support matrix, strict Sphinx and advisory job results.
- [ ] Confirm the protected main branch, release tags, environments with
      deploy access for the named release maintainers only, runner access
      and separately scoped tokens.

### After merging to main

- [ ] Record the reviewed commit SHA and push the immutable candidate tag.
- [ ] Verify the candidate wheel and sdist SHA-256 manifest, clean installs
      and exact-version TestPyPI download and install.
- [ ] Record the candidate pipeline result and check the package page.
- [ ] Push the immutable final tag on the reviewed commit.
- [ ] Verify the final tag's pipeline, retained checksums and TestPyPI upload.
- [ ] Start the protected PyPI publishing job as a named release
      maintainer and record who started it.
- [ ] Verify the published checksums and exact-version PyPI install.
- [ ] Create the GitLab Release with the notes, pipeline link and manifest.
- [ ] Sync release changes to develop through a merge request.
