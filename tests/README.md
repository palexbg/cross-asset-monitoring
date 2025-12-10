# Backend Unit Tests

These tests cover key invariants and sanity checks for the backend engines:

- Return contributions sum to portfolio return
- Volatility risk contributions sum to ~100%
- FX normalization (identity and scaling)
- Rebalance schedule correctness
- Config defaults

## Running the tests

Make sure you have `pytest` installed:

```bash
pip install pytest
```

Then run all tests:

```bash
pytest
```

Or run a specific test file:

```bash
pytest tests/test_risk_contrib.py
```
