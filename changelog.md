# Changelog

All notable changes to this project are documented in this file.

## [v0.1.0] - 2025-12-15
Add LLM-powered sidebar assistant for interactive Q&A on portfolio risk and return.

## [v0.1.0] - 2025-12-10
Initial snapshot and UI/backend hardening.

### Added
- `app/ui/theme.py`: centralized color maps, palettes and shared UI constants used across Streamlit tabs.
- Top-of-page disclaimer added to all Streamlit tabs (clear educational/demo wording).
- `tests/test_config.py` and other small unit tests to assert backend defaults and invariants.

### Changed
- `main.py`: replaced hard-coded demo end date with logic that prefers the latest date in the local CSV cache (`DataConfig.etf_data_path`)
- Streamlit tabs refactored to import shared UI constants from `app/ui/theme.py` (removed duplicated color maps).
- Moved small key-stat builder logic into backend (`backend/perfstats.py`) for better testability and separation of presentation vs logic.

### Fixed / Hardened
- Defensive guards in factor construction to avoid errors on very short date ranges (prevent ValueError from zero-length scaling arrays).
- Updated data loader behavior to persist the small CSV convenience cache when downloading via `YFinanceDataFetcher`.

### Tests & CI
   Please run tests locally with:

```bash
PYTHONPATH=. pytest -q
```

### Notes / Pending
- Performance: duplicated heavy computations for rolling factor exposures are still present across multiple UI tabs. Next improvement: expose cached "latest betas" from the shared analysis context and have tabs consume that cache.
- Warnings: a small number of non-fatal warnings (NaN handling and Streamlit deprecation notices) remain and should be cleaned up for a quieter CI run.

---

(Generated 2025-12-10)
