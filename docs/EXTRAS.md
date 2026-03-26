# Extras / Back-Burner Ideas

Items that would be nice to have but aren't blocking current work.

## Ingestion: `--repair` / `--validate` flag

Scan existing raw tables for gaps (missing date ranges, partial chunks, symbols with
incomplete coverage) and generate a targeted re-ingestion plan. Flow:

1. Walk `raw_option_quotes` and `raw_underlying_bars` checking for contiguous date coverage
2. Cross-reference against chunk manifests to identify missing symbol/date pairs
3. Report gaps with estimated Databento cost to fill them
4. With `--repair`: fetch only the missing data (same cost estimation + confirmation flow)

This is non-trivial because it requires per-symbol date coverage analysis across
potentially millions of rows. Candidate approach: build a coverage bitmap per
(symbol, date) and diff against the manifest.
