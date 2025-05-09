# DHW Probabilistic Integration Tests

This directory contains integration tests for the probabilistic DHW (Domestic Hot Water) methods.

## Test Files

- `objects.csv`: Contains test objects with different configurations for the DHW methods.
- `test_dhw_probabilistic.py`: Contains the integration tests for the DHW methods.

## Test Objects

The `objects.csv` file contains the following test objects:

1. Object with jordan_vajen source and dwelling_size
2. Object with hendron_burch source and occupants
3. Object with iea_annex42 source and household_type
4. Object with jordan_vajen source, dwelling_size, and weekend_activity
5. Object with vdi4655 source and dwelling_size
6. Object with user source and occupants
7. Object with no source but with dwelling_size
8. Object with no source but with occupants
9. Object with no source but with household_type

## Test Cases

The integration tests cover the following test cases:

1. **All Objects Test**: Tests that all objects can be processed without errors.
2. **Individual Objects Test**: Tests each object individually to isolate any issues.
3. **Source Comparison Test**: Tests that different sources produce different results.
4. **Weekend Activity Test**: Tests that weekend activity produces different results for weekdays and weekends.
5. **Parameter Selection Test**: Tests that objects with no source but with different parameters use different methods.
6. **Edge Cases Test**: Tests edge cases like very small or very large values.

## Running the Tests

To run the tests, use the following command from the project root directory:

```bash
pytest tests/integration/dhw_probabilistic
```

## Expected Results

All tests should pass, indicating that the DHW methods are working correctly. The tests verify that:

- All objects can be processed without errors
- The DHW data has the expected structure
- The DHW data has reasonable values
- Different sources produce different results
- Weekend activity produces different results for weekdays and weekends
- Objects with no source but with different parameters use different methods
- Edge cases are handled correctly