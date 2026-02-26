# Release Procedure

This release procedure outlines the steps for managing releases in the GitLab environment.<br>
These symbols help with orientation:
- 🐙 GitLab
- 💠 git (Bash)
- 📝 File
- 💻 Command Line (CMD)
- 🧪 Testing


## Version Numbers

This software follows the [Semantic Versioning (SemVer)](https://semver.org/).<br>
It always has the format `MAJOR.MINOR.PATCH`, e.g. `1.0.0`.

**Note**: This project uses `hatch-vcs` for automatic versioning based on Git tags. The version is automatically derived from the latest Git tag.


## Release Process

### 1. 🧪 Run Complete Test Suite
- **Unit Tests**: Run pytest to ensure all tests pass:
    ```bash
    pytest
    ```
- **Coverage Check**: Verify test coverage meets requirements:
    ```bash
    pytest --cov
    ```
- **Multi-Environment Testing**: Run tox to test across multiple Python versions/environments:
    ```bash
    tox
    ```
- **Build Test**: Verify the package builds correctly:
    ```bash
    uv build
    ```

### 2. 📝 Update the `CHANGELOG.md`
- **File**: Open the CHANGELOG.md file and add a new entry under the `[Unreleased]` section.
- **Commit**: Commit your changes to the changelog, noting all new features, changes, and fixes.
- **Version Entry**: Format the new version entry as follows:
    ```markdown
    ## [1.0.0] Short description - 2024-12-15

    ### Added
    - New feature
    - Another new feature

    ### Changed
    - Change to existing feature

    ### Fixed
    - Bug fix
    ```

### 3. 🐙 Create Release Issue and Branch
- **Template**: Use the `📝Release_Checklist` template for the issue.
- **Issue**: Create a new issue with the title `Release - Version - 1.0.0`.
- **Branch**: Create a release branch:
    ```bash
    git checkout develop  # or main if working directly from main
    git pull
    git checkout -b release/1.0.0
    ```
- **Push**: Push the release branch to GitLab:
    ```bash
    git push --set-upstream origin release/1.0.0
    ```

### 4. 🐙 Create Merge Request and Review
- **Merge Request**: In GitLab, open a merge request (MR) from `release/1.0.0` into `main`.
- **Review**: Assign reviewers to the MR and ensure all tests pass.
- **CI/CD**: Verify that all CI/CD pipelines pass successfully.
- **Merge**: Once approved, merge the MR into main and delete the release branch.

### 5. 💠 Tag the Release (CRITICAL STEP)
- **Checkout main**: Ensure you're on the latest main branch:
    ```bash
    git checkout main
    git pull
    ```
- **Check for commits after merge**: Verify if you need to recreate the tag:
    ```bash
    git log --oneline v1.0.0..HEAD
    ```
    If this shows commits, you need to delete and recreate the tag.

- **Delete existing tag** (if it exists and commits were made after):
    ```bash
    # Delete local tag
    git tag -d v1.0.0
    # Delete remote tag
    git push origin :refs/tags/v1.0.0
    ```
- **Create new tag**: Tag the release pointing to the latest main:
    ```bash
    git tag -a v1.0.0 -m "Release 1.0.0"
    git push origin v1.0.0
    ```

### 6. 🐙 Create GitLab Release
- **GitLab Release Page**: Go to the GitLab project's Releases section and create a new release linked to the v1.0.0 tag.
- **Release Notes**: Add release notes using information from the changelog.

## PyPI Release (Manual Process)

### 1. 🧪 Test on Test-PyPI First
- **Build the package** with the new tag:
    ```bash
    # Clean previous builds
    uv run python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"

    # Build with hatch-vcs (version automatically from tag)
    uv build
    ```
- **Upload to Test-PyPI**:
    ```bash
    uv publish --repository testpypi
    ```
- **Verify on Test-PyPI**: Check [https://test.pypi.org/project/entise/](https://test.pypi.org/project/entise/)
- **Test Installation from Test-PyPI**:
    ```bash
    # Create a clean test environment
    uv venv test_env
    source test_env/bin/activate  # On Windows: test_env\Scripts\activate

    # Install from Test-PyPI
    uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ entise

    # Test basic functionality
    python -c "import entise; print('Import successful')"
    # Run any basic functionality tests here

    # Clean up
    deactivate
    rm -rf test_env
    ```

### 2. 🧪 Final Pre-Release Testing
- **Functional Testing**: Verify the Test-PyPI package works as expected
- **Dependency Check**: Ensure all dependencies are correctly specified and installable
- **Documentation**: Verify that the package metadata and description display correctly on Test-PyPI

### 3. 💻 Publish to PyPI
- **Final Build** (if needed):
    ```bash
    # Only if you haven't built already or made changes
    uv run python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
    uv build
    ```
- **Upload to PyPI**:
    ```bash
    uv publish
    ```
- **Enter Credentials**: Use your PyPI username and password (or API token) when prompted.

### 4. ✅ Post-Release Verification
- **Verify on PyPI**: Check [https://pypi.org/project/entise/](https://pypi.org/project/entise/) to confirm the release.
- **Test Installation from PyPI**:
    ```bash
    # Create fresh environment
    uv venv verify_env
    source verify_env/bin/activate  # On Windows: verify_env\Scripts\activate

    # Install from PyPI
    uv pip install entise==1.0.0  # replace with actual version number

    # Test functionality
    python -c "import entise; print('PyPI installation successful')"

    # Clean up
    deactivate
    rm -rf verify_env
    ```
- **Update Documentation**: Update any documentation that references version numbers.

### 5. 💠 Merge Back to Develop (if using GitFlow)
- **Merge main into develop**:
    ```bash
    git checkout develop
    git pull
    git merge main
    git push
    ```

## Important Notes

- **Automatic Versioning**: This project uses `hatch-vcs` - versions are automatically determined from Git tags. No manual version file updates needed.
- **Tag Timing is Critical**: Always create/recreate the tag AFTER merging to main to ensure it points to the correct commit.
- **Test-PyPI First**: Always test on Test-PyPI before publishing to PyPI to catch any packaging issues.
- **Manual Publishing**: This project uses manual PyPI publishing - no automatic CI/CD deployment.
- **Clean Builds**: Always clean the `dist/` folder before building to avoid uploading old artifacts.
- **Credentials**: Ensure your PyPI credentials are set up correctly (consider using API tokens instead of passwords).
- **Rollback Plan**: If issues arise post-release, you can yank the release on PyPI and fix issues in a patch version.

## Troubleshooting

- **Version not updating**: Make sure your tag points to the latest commit after merging.
- **Test-PyPI upload fails**: Each version can only be uploaded once to Test-PyPI. Use a different version for testing.
- **Import errors after installation**: Check that all dependencies are correctly specified in `pyproject.toml`.
- **Package not found**: Wait a few minutes after uploading - PyPI can take time to process new releases.
