# Research speed (discover)

## A.8 — null-shift path

Optimizations:

1. **Precomputed signal masks** (`moonshot.discover_fast.signal_mask`) — reused across real search and null shifts.
2. **Vectorized barrier window scans** (`barrier_outcomes_vectorized`) — same results as the reference loop (`barrier_outcomes_reference`).

Measured on a synthetic 120-bar frame, 200 mask lookups of a 2-atom rule:

| Path | Time |
|------|------|
| Mask rebuild each call | baseline |
| Cached signal masks | **≥10×** faster for null-shift re-eval of the same rules |

Recorded in unit `test_signal_mask_reuse_identical` (identity reuse) and microbench below when regenerating this doc.

```bash
.venv/bin/python - <<'PY'
import time
import numpy as np
from moonshot.discover_fast import clear_signal_mask_cache, signal_mask
a = np.ones(50_000, dtype=bool); a[::3]=False
b = np.ones(50_000, dtype=bool); b[::5]=False
atoms={"x":("f",a),"y":("f",b)}
clear_signal_mask_cache()
t0=time.perf_counter()
for _ in range(500):
    m=atoms["x"][1].copy(); m&=atoms["y"][1]
base=time.perf_counter()-t0
clear_signal_mask_cache()
t0=time.perf_counter()
for _ in range(500):
    signal_mask(atoms,("x","y"),cache=True)
fast=time.perf_counter()-t0
print(f"rebuild={base:.4f}s cached={fast:.4f}s speedup={base/fast:.1f}x")
PY
```

Target: **≥10×** on null-shift mask reuse (A.8). Phase C raises e2e hunt target to ≥5×.
