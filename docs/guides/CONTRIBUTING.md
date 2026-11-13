<!--SPDX-License-Identifier: MIT-->
<!--Version: v1.0.0-->

# Collaborative Development

## Prerequisites
- An account on the [EnTiSe GitLab host](https://gitlab.lrz.de/).
- [Git](https://git-scm.com/) for version control. [Git How To](https://githowto.com/) and [Git Cheat Sheet](https://training.github.com/downloads/github-git-cheat-sheet.pdf) provide an introduction into working with Git.
- Development Environment Setup:
  - Clone the repository and set up the development environment as described in the
    [README](../../README.rst).

### Optional: Setting Up SSH for GitLab
To simplify authentication, consider setting up SSH keys for GitLab. Follow [GitLab’s SSH key setup guide](https://docs.gitlab.com/ee/user/ssh.html) for instructions.

## Types of interaction
This repository is following the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). <br>
Please be self-reflective and always maintain a good culture of discussion and active participation.

### A. Use
Since the open license allows free use, no notification is required.
However, for the authors it is valuable information who uses the software for what purpose.
Indicators are `Watch`, `Fork` and `Starred` of the repository.
If you are a user, please add your name and details in [USERS.cff](../../USERS.cff).

### B. Comment
You can give ideas, hints or report bugs in issues, in MR, at meetings or other channels.
This is no development but can be considered a notable contribution.
If you wish, add your name and details to [CITATION.cff](../../CITATION.cff).

### C. Contribute and Review
You add code and become an author of the repository.
You must follow the workflow!
Add your name and details to [CITATION.cff](../../CITATION.cff).

### D. Maintain and Release
You contribute and take care of the repository.
You review and answer questions.
You coordinate and carry out the release.
Add your name and details to [CITATION.cff](../../CITATION.cff).

## Workflow

### 1. Agree on scope and where to stop

For substantive work, record the task, affected areas, and verification that
will show it is complete. Check for related issues and merge requests, and
confirm the starting commit on `origin/develop`. Discuss the real design
choices and record the approved approach before implementation. Existing
issues can hold the decision. A local draft can hold it until publication.

For a maintainer-started session, implementation approval covers local
commits, review, and opening or updating the MR once the reviewed change
is ready. Record any narrower endpoint with the design. Use the same
approval for each authorized step. Return to the maintainer for a material
scope change or a blocker that needs their decision. The maintainer decides
when to merge.

### 2. Implement on a topic branch

Use a branch from `develop` and an isolated worktree when sessions share a
checkout. Reuse a GitLab issue that already covers the change. If the issue
will be created at publication, use a temporary local branch name. Once
GitLab assigns a number, use the repository's branch convention, for example
`feature/42-short-description` or `bug/42-short-description`. Keep the
source branch of an existing open MR.

For a behavior fix, first show a focused regression test fails on the base
and then show it passes with the fix. Add an entry under `[Unreleased]` in
[CHANGELOG.md](../../CHANGELOG.md) for changes to outputs or the public API.
Include the issue reference once assigned. Use **Behavior change** where
appropriate. Release preparation handles the version and release heading.

### 3. Verify and review

Run the checks that cover the changed behavior. Unit and integration tests
use local fixtures or mocked services. From the active development environment:

```bash
pytest tests --ignore=tests/installation
pytest tests
```

The first command omits the installation test. The second can reach package
indexes. Use Ruff for lint and formatting and mypy for type checks. When a
check fails, compare with the target and report confirmed baseline failures,
new failures, and any verification gaps. See the [CI guide](CI_CD_Guide.md)
for the checks run by GitLab.

Before publication, review the diff against the current target. Check the
normal caller, compatibility, tests, and any changelog or docs changes. For
tool-assisted review, see the [shared review skill](../../.agents/skills/review-mr/SKILL.md).
Record the reviewed commit and target SHA, material findings and their
resolution, and the exact checks and results. Compare the reviewed SHA with
the publication head. Review later changes and rerun the checks they affect.

### 4. Publish and hand off the MR

Within the approved scope, create or reuse the issue, push only the topic
branch, and open or update the MR against `develop`. Before pushing to an
existing MR branch, check that the MR is still open and that the target and
source are the intended ones. Keep `main` and `develop` for merges and
releases.

Use the matching [MR template](../../.gitlab/merge_request_templates/).
Include the review and verification record and any later changes. Verify
the published MR's source, target, and head and report its link and check
status. Use `Closes #123` when the change resolves the issue. Automatic
closure depends on the GitLab target and project settings. The maintainer
checks issue status when closing out the change.

## Coding Standards

To maintain consistency and readability, please follow the coding standards outlined below when contributing to this project.

### Code Style

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use Ruff for lint
and formatting with the settings in [pyproject.toml](../../pyproject.toml).

### Documentation Style
To ensure consistent and clear documentation, please follow the Google-style docstring format.
This style is straightforward and well-suited for research projects, as it provides structured sections for describing parameters, return values, examples, and more.

#### General Rules
1. **Module-Level Docstring**: Each file should start with a top-level docstring explaining its purpose and any important details.
2. **Class and Function Docstrings**: Each class and function should have a docstring summarizing its behavior. Class docstrings should also document class attributes if applicable.
3. **Attributes**: Document module-level and class attributes with type annotations and descriptions. This can be done inline or in the `Attributes` section.

#### Docstring Format
- **Summary**: Begin with a concise summary of the object’s purpose.
- **Args**: List and describe each parameter, including type annotations and default values.
- **Returns**: Describe the return value(s) of the function, including type annotations. Use `Yields` instead of `Returns` for generator functions. Use `None` if the function returns nothing.
- **Raises**: List any exceptions that the function may raise.
- **Examples**: Provide examples of how to use the function or class.
- **Notes**: Include any additional information that may be helpful.
- **References**: Link to any external resources or references.
- **TODOs**: List any future work or improvements that could be made.

#### Examples of Each Type of Docstring
1. **Module-Level Docstring**
   ```python
   """Brief module description.

    This module demonstrates the Google-style docstring format. It provides
    several examples of documenting different types of objects and methods.

    Attributes:
        module_level_variable1 (int): Module level variables can be documented here.
        module_level_variable2 (str): An example of an inline attribute docstring.
    """
   ```

2. **Function Docstring**
    ```python
    def example_function(param1: int, param2: str = 'default') -> bool:
         """Summary of the function.

         Args:
              param1 (int): The first parameter.
              param2 (str): The second parameter. Defaults to 'default'.

         Returns:
              bool: A boolean value indicating success.

         Raises:
              ValueError: If the parameter is invalid.
         """
    ```

3. **Generator Function Docstring**
    ```python
    def example_generator(param: int) -> Iterator[int]:
         """Summary of the generator function.

         Args:
              param (int): The parameter for the generator.

         Yields:
              int: The next value in the sequence.
         """
    ```

4. **Class Docstring with Attributes**
    ```python
    class ExampleClass:
    """A summary line for the class.

        Attributes:
            attr1 (str): Description of `attr1`.
            attr2 (int, optional): Description of `attr2`, which is optional.
        """

        def __init__(self, attr1: str, attr2: int = 0):
            """Initializes ExampleClass with specified attributes.

            Args:
                attr1 (str): Description of `attr1`.
                attr2 (int, optional): Description of `attr2`. Defaults to 0.
            """
            self.attr1 = attr1
            self.attr2 = attr2
    ```

5. **Exception Docstring**
    ```python
    class CustomError(Exception):
        """Exception raised for specific errors in ExampleClass.

        Args:
            msg (str): Explanation of the error.
            code (int, optional): Error code, if applicable.

        Attributes:
            msg (str): Explanation of the error.
            code (int): Error code.
        """
        def __init__(self, msg: str, code: int = None):
            self.msg = msg
            self.code = code
    ```

6. **Special Methods and Private Methods**
- Document special methods like `__init__`, `__str__`, and `__repr__` in the class docstring or their individual docstrings as you would any other method.
- Private methods (starting with `_`) generally do not require detailed docstrings unless they are complex.

#### Notes
- **Examples**: When adding examples use doctest format where possible. This allows you to test the examples as part of your test suite.
- **Formatting Tips**:
    - Use triple quotes for multi-line docstrings.
    - Use indentation consistently for multiline parameter descriptions.
    - Avoid documenting `self` in `Args` for instance methods, as it is implied.
    - Use `None` for default values instead of `NoneType`.
    - Use `Optional` for optional parameters with default values.
    - For properties, document only in the getter method.

Following this format will ensure all contributors create clear, structured documentation that can be easily read and maintained by others.

## Updating Documentation and Changelog
Keeping documentation up-to-date is essential for ensuring other contributors and users understand new features, changes, and fixes. Please follow these guidelines when updating documentation and the changelog.

### 1. Updating Documentation
- **Main Documentation**: If your changes introduce new features or modify existing functionality, update the relevant sections in the documentation located in the docs/ folder.
  - **Writing Style**: Keep the writing clear, concise, and objective.
  - **Code Examples**: Include code examples where applicable to demonstrate usage.
  - **Building Documentation**: If the project uses MkDocs or Sphinx, preview changes by building the documentation locally:
    ```bash
    mkdocs serve  # For MkDocs
    ```
    ```bash
    make html  # For Sphinx
    ```
- **Docstrings**: Update or add docstrings to any new or modified functions, classes, or modules, following the Google-style guidelines in the Documentation Style section above.

### 2. Updating Changelog
The `CHANGELOG.md` file keeps a record of significant changes to the project and helps users and contributors track updates. Please follow these conventions for maintaining the changelog:

- **Entry Format**:
  - Use the following structure for each new entry:
    ```markdown
    ## [Version] - [YYYY-MM-DD]

    ### Added
    - Brief description of the new feature, referencing the issue or merge request number if applicable.

    ### Changed
    - Description of any modifications to existing features.

    ### Fixed
    - Description of bug fixes or corrections.

    ### Removed
    - Deprecated features or removed functionality.
    ```
- **Examples**:
    - **New Feature**:
        ```markdown
        ## [1.0.0] - 2022-01-01

        ### Added
        - Implemented new feature X (#42).
        ```
    - **Bug Fix**:
        ```markdown
        ## [1.0.1] - 2022-01-02

        ### Fixed
        - Resolved issue with feature X not working correctly (#43).
        ```
- **Semantic Versioning**: Follow [Semantic Versioning](https://semver.org/) guidelines when updating the version number. Increment the version based on the impact of the changes:
  - **Major**: Breaking changes or significant new features.
  - **Minor**: New features or enhancements that are backward-compatible.
  - **Patch**: Bug fixes or minor improvements that do not affect compatibility.

### 3. Reviewing Documentation and Changelog Before Submission
- **Double-Check**: Ensure all new functionality is covered in the documentation, and verify that changelog entries are clear and concise.
- **Consistency**: Use consistent terminology and formatting across all documentation and changelog entries.
- **Preview Changes**: When possible, preview the documentation and changelog to ensure they are visually correct and well-organized.

By following these guidelines, you’ll help maintain high-quality, organized documentation that benefits all users and contributors.
