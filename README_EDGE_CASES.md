# CRT Edge Cases

This pack adds **deterministic unit tests**, **property-based tests**, and a **JSON generator** for CRT edge cases.

## What’s included

- `tests/test_crt_edge_cases_exhaustive.py` — explicit, table-driven sequences that validate CRT behavior for:
  - BULL/BEAR C2 basics,
  - C3 confirmations by `high` and by `close` with `confirm_within` windows,
  - equality vs strict edges at the C1 boundaries,
  - dual sweep handling (skip vs allow both),
  - gap up/down scenarios,
  - no-wick bars,
  - narrow C1 range (zero-width),
  - robustness to `NaN` rows,
  - gating flags: `require_midline`, `strict_vs_c1open`,
  - `directions` filters,
  - confirm method case-insensitivity (at least `"CLOSE"`).

- `tests/test_crt_properties.py` — **Hypothesis**-based invariants that fuzz around boundaries and verify:
  - no look-ahead is required for C2 setup detection,
  - dual sweep toggle semantics,
  - close-inside vs outside invariants,
  - confirm method casing does not change outcomes for `"close"`.

- `tools/gen_crt_edge_cases.py` — generates a JSON catalog of edge-case sequences for manual/visual testing or for feeding other test harnesses.

## How to run

```bash
# Install test deps (Python 3.11)
pip install -U pytest hypothesis pandas numpy

# Run tests
pytest -q

# Generate JSON with edge-case sequences
python tools/gen_crt_edge_cases.py > crt_edge_cases.json
```

## CRT assumptions enforced

- **C1 = previous single candle** (N=1): `C1 = [Low[i-1], High[i-1]]`.
- **C2** strict sweep of one side + **Close back inside** (inclusive).
- **C3** confirmation by `high` or `close` within `confirm_within` bars after C2.
- **No look-ahead bias** for setup detection.
- **Float tolerance** conceptually `EPS=1e-9` in tests; implementation may use plain comparisons — tests choose values that are numerically unambiguous.

## Notes

- If your `crt_scan` exposes `skip_dual_sweep`, tests will assert both skip/allow behaviors. If not, those tests will be skipped gracefully.
- Some repos treat `confirm_method` case-sensitively for `"high"`. The test suite includes a **best-effort** case that will **xfail** if your implementation is case-sensitive.
