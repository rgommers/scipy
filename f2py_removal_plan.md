# Plan: Remove f2py dependency from scipy/linalg

## Context

f2py is only used in `scipy/linalg/` to generate four extension modules (`_fblas`, `_flapack`, `_fblas_64`, `_flapack_64`) from ~7,400 lines of `.pyf.src` template files. These provide the Python-callable public API for BLAS/LAPACK routines via `get_blas_funcs()` / `get_lapack_funcs()` and bare imports like `from scipy.linalg.blas import dgemm`.

Meanwhile, **Cython wrappers already exist** (`cython_blas` with 149 routines, `cython_lapack` with 1497 routines) that cover a superset of the f2py-wrapped routines. These are `cdef` pointer-based functions usable only from Cython. They are already used extensively across scipy (cluster, optimize, interpolate, etc.).

The goal is to eliminate the f2py build dependency entirely.

## Current f2py footprint

| Component | Location |
|-----------|----------|
| `.pyf.src` files (15 files, ~7400 lines) | `scipy/linalg/fblas*.pyf.src`, `flapack*.pyf.src` |
| Build infrastructure | `scipy/meson.build` (f2py detection, fortranobject_dep, f2py_gen, f2py_ilp64_opts) |
| Build targets | `scipy/linalg/meson.build` (4 extension modules) |
| Template processor | `scipy/tools/generate_f2pymod.py` |
| Public API | `scipy/linalg/blas.py` (imports `_fblas`/`_fblas_64`), `scipy/linalg/lapack.py` (imports `_flapack`/`_flapack_64`) |
| Signature generator (dev-only) | `scipy/linalg/_cython_signature_generator.py` (uses `numpy.f2py.crackfortran`) |

**No other scipy subpackage uses f2py.** `fortranobject_dep` and `f2py_gen` are only consumed by `scipy/linalg/meson.build`.

## What f2py wrappers provide (that raw Cython `cdef` wrappers don't)

The `.pyf.src` files encode significant wrapper logic beyond just calling the Fortran function:

1. **NumPy array I/O** - accepts/returns ndarray, not raw pointers
2. **Automatic Fortran-order conversion** - C-contiguous arrays transparently converted
3. **Automatic dtype casting** - e.g., float64 passed to `s*` routine triggers conversion
4. **Dimension inference** - `m = shape(a,0)`, hidden from Python API
5. **Input validation** - `check()` directives for bounds/shape
6. **Default values** - optional parameters with defaults
7. **Custom callstatements** - integer-to-char mapping (`trans_a ? "T" : "N"`), index adjustments (1-based to 0-based pivots), post-processing loops
8. **Copy/overwrite semantics** - `intent(in,out,copy)` with `overwrite_*` flags
9. **Workspace allocation** - `intent(cache,hide)` for work arrays
10. **Output renaming** - `intent(out, out=lu)`

## Options

### Option A: Generated Cython `def` wrappers (recommended)

Create a new code generator that produces Cython `.pyx` files with `def` functions wrapping the existing `cython_blas`/`cython_lapack` `cdef` functions. The generated `def` functions replicate all the f2py wrapper semantics listed above.

**Pros:**
- Zero-overhead calls to existing Cython `cdef` functions
- Build system already has Cython infrastructure
- Can be auto-generated from a specification (parsed from `.pyf.src` or a new format)
- Full control over behavior, no external dependency

**Cons:**
- Large effort: ~100 BLAS + ~300 LAPACK routines, many with unique callstatement logic
- Need to parse/translate ~7,400 lines of f2py-specific DSL
- Risk of subtle behavioral differences in edge cases

**Approach:**
1. Parse `.pyf.src` files (after template expansion) to extract per-routine specs: arguments, intents, defaults, checks, callstatements, dimension dependencies
2. Write a generator (`_generate_python_wrappers.py`) that emits Cython `def` functions
3. Build as `_pyblas.pyx` / `_pylapack.pyx` (+ `_64` variants via templating)
4. Update `blas.py` / `lapack.py` to import from new modules
5. Remove all f2py infrastructure

### Option B: Hand-written Cython wrappers

Write each wrapper function by hand, referencing the `.pyf.src` specifications.

**Pros:** Full control, clean code
**Cons:** Enormous effort for 400+ routines, very error-prone, hard to maintain

### Option C: Pure Python wrappers using ctypes/cffi

Write Python functions that call BLAS/LAPACK via ctypes or cffi.

**Pros:** No Cython needed for this layer
**Cons:** Higher call overhead, more complex memory management, doesn't leverage existing Cython infrastructure

### Option D: Keep f2py but vendor it / use it as build-time only

Bundle a minimal f2py code generator rather than depending on NumPy's.

**Pros:** Minimal code changes
**Cons:** Doesn't actually remove the dependency, just internalizes it; maintenance burden

## Recommendation: Option A

Option A is the most practical path. The key insight is that the `.pyf.src` files are a machine-readable specification — we can parse them systematically rather than rewriting 400+ wrappers by hand.

## Implementation phases

### Phase 1: Build the specification parser
- Parse expanded `.pyf.src` files to extract structured routine specs
- Could reuse parts of `tools/generate_f2pymod.py` for template expansion
- Output: JSON or Python dict per routine with args, intents, defaults, checks, callstatement, etc.
- **Critical files:** `scipy/linalg/fblas_l1.pyf.src`, `fblas_l2.pyf.src`, `fblas_l3.pyf.src`, `flapack_*.pyf.src`

### Phase 2: Build the Cython wrapper generator
- Takes parsed specs, emits `.pyx` files with `def` functions
- Each generated function: validates inputs, converts arrays, infers dimensions, calls `cython_blas.*` / `cython_lapack.*`, returns results
- Handle all intent patterns: `in`, `out`, `in,out,copy`, `hide`, `cache,hide`
- Translate callstatement expressions to Cython code
- **Model after:** `scipy/linalg/_generate_pyx.py` (existing Cython wrapper generator)

### Phase 3: BLAS wrappers first
- Generate `_pyblas.pyx` covering Level 1-3 BLAS (~100 routines)
- Wire into `meson.build` alongside old f2py modules
- Validate: run `test_fblas.py` and `test_blas.py` against new module
- **Critical files:** `scipy/linalg/meson.build`, `scipy/linalg/blas.py`

### Phase 4: LAPACK wrappers
- Generate `_pylapack.pyx` (~300 routines)
- Handle complex callstatements (index adjustments, char mapping, workspace queries)
- Validate: run `test_lapack.py` against new module

### Phase 5: ILP64 variants
The generated `.pyx` files should use `blas_int` throughout (not `int` or `int64_t` directly). The LP64 vs ILP64 difference is **only** the typedef of `blas_int` and the symbol mangling macros in the C headers — exactly how the existing `cython_blas`/`cython_lapack` already work.

**Approach:** Generate `_pyblas.pyx.tmpl` and `_pylapack.pyx.tmpl` as templates. At build time, the generator (extending `_generate_pyx.py` or a sibling script) produces both LP64 and ILP64 variants by:
- Setting `ctypedef int blas_int` (LP64) vs `ctypedef int64_t blas_int` (ILP64) in the `.pxd`
- Using `BLAS_FUNC()` macro consistently in the generated C headers
- The `.pyx` source itself is identical — it just uses `blas_int` everywhere

This mirrors the existing mechanism in `_generate_pyx.py` which already takes an `--ilp64` flag and produces different `cython_blas.pxd` / `cython_lapack.pxd` accordingly. The new Python-facing wrappers simply `cimport` from `cython_blas` / `cython_lapack` and inherit the correct `blas_int` typedef.

No separate hand-maintained `_64` source files needed.

### Phase 6: Cutover and cleanup
- Update `blas.py` / `lapack.py` to import from new modules
- Remove `.pyf.src` files, `tools/generate_f2pymod.py`
- Remove f2py infrastructure from `scipy/meson.build` (f2py detection, fortranobject_dep, f2py_gen, f2py_ilp64_opts, int64_f2cmap)
- Remove f2py targets from `scipy/linalg/meson.build`
- Update docstrings referencing f2py behavior

### Note on `_cython_signature_generator.py`
This dev-only script uses `numpy.f2py.crackfortran` to parse Fortran source and generate `cython_*_signatures.txt`. It is **not run during builds** — the signature files are checked in. This can be left as-is initially (NumPy will keep shipping crackfortran), or rewritten later with a simple Fortran parser since it only needs function name, return type, and argument names/types.

## Verification
- Run full `scipy/linalg/` test suite: `python -m pytest scipy/linalg/tests/`
- Specifically: `test_fblas.py`, `test_blas.py`, `test_lapack.py`, `test_cython_blas.py`, `test_cython_lapack.py`
- Run higher-level tests that exercise BLAS/LAPACK through `get_*_funcs`: `test_basic.py`, `test_decomp.py`, `test_solvers.py`
- Benchmark critical routines (dgemm, dgesv, dsyev) to check for performance regressions
- Verify ILP64 builds still work
- Verify `from scipy.linalg.blas import dgemm` still works (bare imports)
