# Guidelines for TS_Creator

## Table of Contents
1. [Introduction](#introduction)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [Coding Standards](#coding-standards)
6. [Method Implementation Guidelines](#method-implementation-guidelines)
7. [Development Workflow](#development-workflow)
8. [Testing Strategy](#testing-strategy)
9. [Documentation Requirements](#documentation-requirements)
10. [Performance Considerations](#performance-considerations)
11. [Security Guidelines](#security-guidelines)
12. [Contribution and Code Review Process](#contribution-and-code-review-process)
13. [Version Control and Branching Strategy](#version-control-and-branching-strategy)
14. [Error Handling and Logging Standards](#error-handling-and-logging-standards)
15. [Dependency Management](#dependency-management)

## Introduction

TS_Creator is a modular framework designed for generating time series data for energy systems. This document provides comprehensive guidelines for developing, maintaining, and contributing to the project, ensuring consistency, quality, and adherence to best practices across the project.

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **pandas**: For data manipulation and time series operations
- **joblib**: For parallel processing
- **tqdm**: For progress tracking

### Development Tools
- **pytest**: For testing
- **flake8**: For code linting
- **black**: For code formatting
- **mypy**: For static type checking
- **pre-commit**: For automated code quality checks

## Project Structure

The project follows a modular architecture:

```
/
├── entise/                # Main package containing the core functionality
│   ├── core/              # Core classes and functionality
│   │   ├── base.py        # Base classes for methods
│   │   ├── generator.py   # Time series generation engine
│   │   └── ...
│   ├── methods/           # Implementation of various time series generation methods
│   │   ├── auxiliary/     # Helper methods
│   │   ├── dhw/           # Domestic hot water methods
│   │   ├── electricity/   # Electricity consumption methods
│   │   ├── hvac/          # Heating, ventilation, and air conditioning methods
│   │   ├── mobility/      # Mobility and transportation methods
│   │   ├── occupancy/     # Occupancy modeling methods
│   │   └── multiple/      # Methods that generate multiple time series types
│   ├── constants/         # Constant values and definitions
│   └── __init__.py
├── docs/                  # Documentation files
│   └── source/            # Source files for documentation
├── examples/              # Example scripts and notebooks
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── CONTRIBUTING.md        # Contribution guidelines
├── README.rst             # Project overview
└── CHANGELOG.md           # Record of changes
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git for version control
- Understanding of time series data and energy systems concepts

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   # For Windows
   venv\Scripts\activate
   # For Linux/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Coding Standards

### Python Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code formatting
- Use 4 spaces for indentation
- Maximum line length of 79 characters
- Use meaningful variable and function names that describe their purpose
- Keep functions and methods small and focused on a single responsibility

### Type Annotations
- Use type hints for all function parameters and return values
- Use typing module for complex types (List, Dict, etc.)
- Example:
  ```python
  from typing import List, Dict, Any

  def process_data(input_data: Dict[str, Any], options: List[str] = None) -> pd.DataFrame:
      # Function implementation
  ```

### Documentation Style
- Use Google-style docstrings for all classes, methods, and functions
- Include examples where appropriate
- Document parameters, return values, and exceptions

Example:
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

    Examples:
        >>> example_function(42, 'test')
        True
    """
```

## Method Implementation Guidelines

When implementing new time series generation methods:

1. **Inherit from the Base Class**: All methods should inherit from the `Method` base class in `entise.core.base`.

2. **Define Required Attributes**:
   - `name`: A unique identifier for the method
   - `types`: List of valid time series types this method can generate
   - `required_keys`: List of required parameters
   - `optional_keys`: List of optional parameters
   - `required_timeseries`: List of required input time series
   - `optional_timeseries`: List of optional input time series

3. **Implement the `generate` Method**:
   ```python
   def generate(self, obj: dict, data: dict, ts_type: str) -> Dict[str, Any]:
       """
       Generate time series data.

       Args:
           obj (dict): Object parameters
           data (dict): Input data
           ts_type (str): Type of time series to generate

       Returns:
           Dict[str, Any]: Generated data with 'summary' and 'timeseries' keys
       """
   ```

4. **Return Standardized Output**:
   - Return a dictionary with 'summary' and 'timeseries' keys
   - Ensure time series data is in pandas DataFrame format

5. **Follow Best Practices**:
   - Ensure methods are deterministic when using the same seed
   - Document any assumptions or limitations in docstrings
   - Validate input parameters before processing
   - Handle edge cases gracefully (e.g., missing data, invalid inputs)

Example implementation:

```python
from entise.core.base import Method
from typing import Dict, Any, List
import pandas as pd


class ExampleMethod(Method):
   name = "example_method"
   types = ["electricity", "heat"]
   required_keys = ["parameter1", "parameter2"]
   optional_keys = ["optional_parameter"]
   required_data = ["input_timeseries"]
   optional_data = ["optional_timeseries"]

   def generate(self, obj: dict, data: dict, ts_type: str) -> Dict[str, Any]:
      """Generate example time series data.

      Args:
          obj (dict): Object parameters
          data (dict): Input data
          ts_type (str): Type of time series to generate

      Returns:
          Dict[str, Any]: Generated data
      """
      # Implementation logic

      # Return standardized output
      return {
         "summary": {"total_value": 100, "peak_value": 10},
         "timeseries": pd.DataFrame({"value": [1, 2, 3, 4, 5]})
      }
```

## Development Workflow

### Basic Workflow
1. **Open an issue** to discuss new features, bugs, or changes
2. **Create a new branch** for each feature or bug fix based on an issue
3. **Write code** and **tests** for the new feature or bug fix
4. **Run tests** to ensure the code works as expected
5. **Create a merge request** to merge the new feature or bug fix into the develop branch
6. **Review the code** and **tests** in the merge request
7. **Merge the merge request** after approval

### Detailed Process
1. **Describe the issue on GitLab**
   - Create an issue in the GitLab repository
   - The issue title should describe the problem you will address
   - Make a checklist for all needed steps if possible

2. **Create a feature branch**
   - Branch naming convention: `type-issue_number-short_description`
   - Types: `feature`, `bugfix`, `hotfix`, `release`
   - Example: `feature-42-add-new-method`

3. **Implement changes**
   - Follow coding standards and guidelines
   - Write tests for new functionality
   - Update documentation as needed
   - Add your changes to the CHANGELOG.md

4. **Commit your changes**
   - Write meaningful commit messages
   - Include the issue number in the commit message
   - Example: `Add new electricity consumption method #42`

5. **Submit a merge request**
   - Direct the merge request to the develop branch
   - Add the line `Close #<issue-number>` in the description
   - Assign a reviewer and get in contact

6. **Code review**
   - Address feedback from reviewers
   - Make necessary changes
   - Get approval for your changes

7. **Merge and close**
   - Merge the feature branch into develop after approval
   - Close the issue with a summary of changes

## Testing Strategy

### Writing Tests
- Write unit tests for all business logic functions
- Implement integration tests for key components
- Use pytest as the testing framework
- Aim for 80%+ code coverage
- Use fixtures for test data setup
- Test edge cases and error conditions

### Running Tests
```bash
pytest tests/
```

### Code Quality Checks
```bash
flake8 entise/
black entise/
mypy entise/
```

### Pre-commit Hooks
To automate code quality checks before each commit, we use pre-commit hooks:

```bash
pre-commit run --all-files
```

## Documentation Requirements

- Maintain up-to-date API documentation
- Write clear README files for each major component
- Document method requirements and outputs
- Create user guides and examples
- Update documentation with each significant change

### Documentation Structure
1. **Module-Level Docstring**: Each file should start with a top-level docstring explaining its purpose
2. **Class and Function Docstrings**: Each class and function should have a docstring summarizing its behavior
3. **Examples**: Include examples of how to use the function or class

### Building Documentation
The documentation is built using Sphinx:

```bash
cd docs
make html
```

## Performance Considerations

- Use vectorized operations with pandas where possible
- Leverage parallel processing for independent operations using joblib
- Implement caching for frequently accessed data
- Use background tasks for long-running operations
- Optimize memory usage for large datasets
- Consider the following when implementing methods:
  - Time complexity of algorithms
  - Memory usage for large datasets
  - Parallelization opportunities
  - Caching strategies for repeated calculations

## Security Guidelines

- Never store sensitive information in code repositories
- Use environment variables for configuration
- Validate all user inputs
- Regularly update dependencies to address security vulnerabilities
- Follow the principle of least privilege
- Implement proper error handling to avoid information leakage

## Contribution and Code Review Process

### Pull Request Guidelines
- Use descriptive titles and descriptions
- Reference related issues
- Include tests for new functionality
- Update documentation as needed
- Update CHANGELOG.md with your changes

### Code Review Standards
- At least one approval required before merging
- Code author cannot approve their own PR
- Automated tests must pass
- Code style checks must pass

### Merge Criteria
- All discussions must be resolved
- CI pipeline must pass
- Documentation must be updated

## Version Control and Branching Strategy

### Branch Naming Conventions
- `feature/short-description`: For new features
- `bugfix/issue-number-description`: For bug fixes
- `hotfix/critical-issue-description`: For critical fixes
- `release/vX.Y.Z`: For release preparation

### Commit Message Format
```
[Component] Short description (50 chars max)

Detailed explanation if necessary. Wrap at 72 characters.
Include motivation for change and contrast with previous behavior.

Refs #123
```

### Version Tagging
- Follow Semantic Versioning
- Tag all releases in Git
- Include release notes with each tag

## Error Handling and Logging Standards

### Error Hierarchy
- Define specific exception types for different error scenarios
- Use custom exceptions for domain-specific errors
- Inherit from appropriate base exception classes

### Logging Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened
- **ERROR**: Due to a more serious problem, the software couldn't perform some function
- **CRITICAL**: A serious error indicating the program may be unable to continue running

### Log Format
```
{timestamp} [{level}] {module}: {message} {context}
```

### Contextual Logging
- Include relevant context with each log entry (object ID, method name, etc.)
- Structure logs for easy parsing and analysis

## Dependency Management

### Dependency Documentation
- Document all dependencies in requirements.txt with pinned versions
- Include comments explaining why each dependency is needed

### Dependency Updates
- Schedule regular dependency updates
- Test thoroughly after updates
- Document any breaking changes

### Dependency Approval Process
- New dependencies must be approved by the team
- Consider security, maintenance status, and license compatibility

## Best Practices

### Time Series Generation
- Ensure methods are deterministic when using the same seed
- Document any assumptions or limitations
- Validate input parameters before processing
- Handle edge cases gracefully

## Troubleshooting

### Common Issues
- **Missing Dependencies**: Ensure all requirements are installed with `pip install -r requirements.txt`
- **Type Errors**: Check type annotations and input validation in your method implementations
- **Performance Issues**: Consider using profiling tools like `cProfile` or `line_profiler` to identify bottlenecks
- **Method Registration Failures**: Verify that your method class has the required attributes (`name`, `types`, etc.)
- **Data Format Errors**: Ensure your time series data is in the correct pandas DataFrame format
- **Seed Inconsistency**: When using random number generation, set a fixed seed for reproducibility

### Debugging Tips
- Use logging at appropriate levels to trace execution flow
- Add debug print statements to inspect variable values
- Test methods in isolation before integrating them
- Validate input data structure before processing
- Check for NaN or missing values in your data

### Troubleshooting Decision Tree
1. **Installation Issues**
   - Check Python version compatibility
   - Verify all dependencies are installed
   - Try creating a fresh virtual environment

2. **Method Implementation Issues**
   - Verify class inheritance from `Method`
   - Check all required attributes are defined
   - Ensure the `generate` method is properly implemented

3. **Runtime Errors**
   - Check input data format and structure
   - Verify parameter types match expectations
   - Look for edge cases in your data

### Getting Help
- Check existing issues on the repository
- Consult the documentation
- Run the examples to understand expected behavior
- Reach out to the maintainers with a minimal reproducible example
