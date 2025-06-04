
# Changelog

All notable changes to this project will be documented in this file.
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]


## [0.2.0] New architecture, methods and packaging - 2025-06-04
### Added
- Added a simple RC model for HVAC time series (`#12`, `!6`)
- Added simpler functionality for dependent methods (`#18`, `!12`)
- Added basic documentation of the package (`#13`, `!8`)
- Added a dhw method based on the method by Jordan et. al. in DHWCalc (`#16`, `!15`)
- Added a PV generation method based on pvlib (`#29`, `!22`)
- Added direct access methods to provide two ways for interacting with the tool (batch & singular) (`#31`, `!26`)
- Converted the project into a proper Python package with modern tooling (`#32`, `!27`)
- Added support for automatic versioning using git tags with hatch-vcs (`#32`, `!27`)
- Added CI/CD configurations for both GitHub and GitLab (`#32`, `!27`)
- Added support for Python 3.10-3.13 (`#32`, `!27`)

### Changed
- Restructured entire architecture towards a pipeline- and strategy-based approach to make methods more flexible (`#18`, `!12`)
- Replaced setuptools with hatchling for modern build system (`#32`, `!27`)
- Replaced flake8, black, and isort with ruff for faster linting and formatting (`#32`, `!27`)
- Replaced pip with uv for faster dependency management (`#32`, `!27`)
- Updated pvlib to version 0.12.0 with different (IANA-based) timezone handling (`#32`, `!27`)
- Updated Python version requirement to 3.10 or newer (`#32`, `!27`)

## [0.1.0] Initial Release - 2024-11-04
### Added
- Initial setup of the project with initial architecture (no methods added yet)

---

# Guidelines for Updating the Changelog
## [Version X.X.X] - YYYY-MM-DD
### Added
- Description of newly implemented features or functions, with a reference to the issue or MR number if applicable (e.g., `#42`).

### Changed
- Description of changes or improvements made to existing functionality, where relevant.

### Fixed
- Explanation of bugs or issues that have been resolved.

### Deprecated
- Note any features that are marked for future removal.

### Removed
- List of any deprecated features that have been fully removed.

---

## Example Entries

- **Added**: `Added feature to analyze time-series data from smart meters. Closes #10.`
- **Changed**: `Refined energy demand forecast model for better accuracy.`
- **Fixed**: `Resolved error in database connection handling in simulation module.`
- **Deprecated**: `Marked support for legacy data formats as deprecated.`
- **Removed**: `Removed deprecated API endpoints no longer in use.`

---

## Versioning Guidelines

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
- **Major** (X): Significant changes, likely with breaking compatibility.
- **Minor** (Y): New features that are backward-compatible.
- **Patch** (Z): Bug fixes and minor improvements.

**Example Versions**:
- **[2.1.0]** for a backward-compatible new feature.
- **[2.0.1]** for a minor fix that doesn’t break existing functionality.

## Best Practices

1. **One Entry per Change**: Each update, bug fix, or new feature should have its own entry.
2. **Be Concise**: Keep descriptions brief and informative.
3. **Link Issues or MRs**: Where possible, reference related issues or merge requests for easy tracking.
4. **Date Each Release**: Add the release date in `YYYY-MM-DD` format for each version.
5. **Organize Unreleased Changes**: Document ongoing changes under the `[Unreleased]` section, which can be merged into the next release version.
