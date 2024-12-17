tests/
├── unit/                               # Unit tests for individual classes and functions
│   ├── __init__.py
│   ├── test_base.py                    # Tests the base class for the methods
│   ├── test_dependency_resolver.py     # Tests the dependency resolver (to ensure it is executed in the right order)
│   ├── test_registry.py                # Tests the registry that registers all the methods
│   ├── test_result_collector.py        # Tests the result collector
│   ├── test_timeseries_generator.py    # Tests the timeseries generator
│   └── test_validation.py              # Tests the validation of the input data
├── integration/                        # Integration tests for full workflows
│   ├── __init__.py
│   └── test_end_to_end.py              # Tests the full workflow from start to finish
├── fixtures/                           # Reusable test data and setup
│   ├── __init__.py
│   ├── timeseries_fixtures.py          # Fixtures for timeseries data
│   └── object_fixtures.py              # Fixtures for objects
├── utils/                              # Utility functions for testing
│   ├── __init__.py
│   └── mock_methods.py                 # Mock methods for testing
├── __init__.py
├── conftest.py                         # Configuration for pytest
└── README.txt                          # Structure of the tests directory
