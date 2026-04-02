# f2py Removal Session Summary

## Primary Request and Intent

The user (rgommers, scipy maintainer) requested a comprehensive removal of the f2py dependency from scipy/linalg. The goal is to replace the f2py-generated `_fblas`/`_flapack` extension modules with new Cython-generated `_pyblas`/`_pylapack` modules that wrap the existing `cython_blas`/`cython_lapack` cdef functions. The work includes:

- Analyzing all f2py usage in scipy (found to be only in scipy/linalg)
- Creating a plan document (`f2py_removal_plan.md`)
- Building a .pyf.src parser and Cython wrapper generator
- Integrating into the Meson build system
- Wiring `blas.py` and `lapack.py` to use the new modules
- Removing f2py build infrastructure
- Making all tests pass
- Debugging specific segfaults and test failures as reported

## Key Technical Concepts

- f2py interface files (.pyf.src) with template expansion (`<prefix=s,d,c,z>`)
- Cython `cdef` vs `def` functions and typedef compatibility
- BLAS/LAPACK calling conventions (Fortran-order arrays, pointer args, 1-based indexing)
- f2py callstatement semantics (pre/post-call processing, character mapping, workspace queries)
- LP64 vs ILP64 integer variants (`blas_int` typedef)
- Meson build system (custom_target, generator, extension_module)
- Python 3 bytes vs str handling (`b"V"[0]` returns int 86, not char 'V')
- f2py intent system (in, out, copy, hide, cache, overwrite)
- Callback function handling for gees/gges routines (eigenvalue selection)
- Workspace query mechanism (lwork=-1 convention)
- Pivot index conversion (0-based Python <-> 1-based Fortran)
- Batch operation framework (`_apply_over_batch`)

## Files and Code Sections

### `scipy/linalg/_pyf_parser.py` (~880 lines)

- Parses .pyf.src template files into structured Python dicts
- Handles template expansion (vendored from tools/generate_f2pymod.py)
- Extracts routine specs: args, intents, defaults, checks, dimensions, callstatements
- Key fixes: balanced paren extraction, standalone intent lines, end-subroutine regex for no-space cases, inline Fortran comment stripping
- Parses 150 BLAS + 629 LAPACK routines

### `scipy/linalg/_generate_pywrappers.py` (~2200+ lines)

Main code generator producing Cython `def` wrappers. Key functions:

- `_get_python_args()` - determines visible Python args from pyf spec
- `_generate_wrapper_function()` - generates one wrapper function
- `_generate_lwork_wrapper()` - generates workspace query functions
- `_generate_gees_gges_wrappers()` - hand-written callback routines
- `_generate_extra_blas_wrappers()` - hand-written missing BLAS (dspr2, chpr2, zhpr2)
- `_generate_checks()` - input validation from check() directives + dimension checks
- `_generate_post_call()` - pivot decrement, scalar decrement, array copy
- `_translate_f2py_expr()` - translates f2py expressions to Python
- `_translate_char_ternary()` - 5 patterns for integer->char mapping
- `_build_call_args()` - builds LAPACK/BLAS call argument string
- `_build_return()` - determines return value order (arg_names order, info last)
- `_load_cdef_signatures()` - loads function signatures from cython_*_signatures.txt
- `_call_select2()/_call_select3()` - callback arg-count retry using inspect

### `scipy/linalg/meson.build`

- Removed: _fblas, _flapack, _fblas_64, _flapack_64 build targets
- Added: _pyblas, _pylapack custom_target + extension_module
- Added: `linalg_blas_lapack_cython_gen` generator depending on both cython_blas_pxd and cython_lapack_pxd

### `scipy/meson.build`

- Removed: f2py detection, fortranobject_dep, f2py_gen, f2py_freethreading_arg, f2py_tls_define, f2py_ilp64_opts, incdir_f2py, int64.f2cmap generation

### `meson.build` (top-level)

- Removed: generate_f2pymod program detection

### `scipy/linalg/blas.py` / `scipy/linalg/lapack.py`

- Changed imports: `_fblas` -> `_pyblas as _fblas`, `_flapack` -> `_pylapack as _flapack`
- Updated docstrings removing f2py references

### `scipy/linalg/_decomp_cholesky.py`

- `cho_factor`: kept lower tiling for batch compatibility
- `_cho_solve`: added `int(lower)` -> `bool(np.asarray(lower).flat[0])` conversion for batched lower array

### `scipy/linalg/tests/test_fblas.py`

- Changed: `from scipy.linalg import _pyblas as fblas`
- Changed: all `blas_func = fblas.xxx` -> `blas_func = staticmethod(fblas.xxx)` (Cython function binding fix)

### `scipy/linalg/tests/test_blas.py`

- Changed: `from scipy.linalg import _pyblas as fblas`, `FBLAS_ERROR = fblas.error`

### `scipy/linalg/tests/test_lapack.py`

- Changed: `from scipy.linalg import _pylapack as flapack`

### Deleted files

- `tools/generate_f2pymod.py`
- `scipy/linalg/fblas_64.pyf.src`, `scipy/linalg/flapack_64.pyf.src`
- `scipy/_build_utils/int64.f2cmap.in`

### Retained .pyf.src files (13 files, used as input to new generator)

- `fblas.pyf.src`, `fblas_l1.pyf.src`, `fblas_l2.pyf.src`, `fblas_l3.pyf.src`
- `flapack.pyf.src`, `flapack_user.pyf.src`, `flapack_gen.pyf.src`, `flapack_gen_banded.pyf.src`, `flapack_gen_tri.pyf.src`, `flapack_sym_herm.pyf.src`, `flapack_pos_def.pyf.src`, `flapack_pos_def_tri.pyf.src`, `flapack_other.pyf.src`

## Errors and Fixes

### Cython typedef compatibility (`float complex *` vs `s *`)

- Used aliased imports: `from scipy.linalg.cython_blas cimport s as cy_s, d as cy_d, c as cy_c, z as cy_z`
- Used cdef signature types from cython_*_signatures.txt for correct pointer casts

### Character arguments (bytes vs str)

- `b"NTC"` bytes literals with `(<char *>b"NTC" + trans)` for char* pointer arithmetic
- Added `if isinstance(arg, str): arg = arg.encode()` conversion
- Fixed `b"V"[0] == 'V'` -> always False in Python 3; translated to `var == b"V"` comparisons

### Pre-call pivot increment missing (ROOT CAUSE of integrate/lobpcg segfaults)

- f2py callstatements like `{F_INT i;for(i=0;i<n;++piv[i++]);(*f2py_func)(...);for(i=0;i<n;--piv[i++]);}` have PRE-call `++piv[i++]` converting 0-based->1-based
- Generator only handled post-call `--piv[i++]`, not pre-call `++piv[i++]`
- Fix: rejoin pre_call parts with `;`, find all `++arr[` patterns in rejoined string
- Also handle multi-array loops: `++ipiv[i],++jpiv[i++]`

### lwork sentinel confusion (-1 used for both "use default" and "LAPACK query")

- Changed sentinel for lwork/liwork/lrwork from -1 to 0
- -1 now passes through to LAPACK for workspace queries (safecall mechanism)

### Return order wrong (e.g., dsyevd returning v,w,info instead of w,v,info)

- Fixed: follow arg_names order for output args, with info always last

### Hidden integer array cdef collision (iwork declared as blas_int scalar, then assigned array)

- Skip hidden args with dimension from scalar cdef declarations

### 1D array handling

- Arrays without explicit intent default to intent(in) for dtype conversion
- 0D scalar->(1,1) reshape, 1D->(-1,1) reshape for 2D parameters
- Track `_was_1d_*` flag and squeeze back on return

### Empty intent arrays

- Arrays with no intents (e.g., `syr`'s `x`) weren't getting dtype conversion -> garbage data passed to BLAS. Fixed by defaulting empty intents to `['in']`.

### Parser end-subroutine regex

- `end subroutineFOO` (no space) from template expansion wasn't matched. Changed `\b` to `\w*` in regex. Recovered 4 BLAS + 28 LAPACK routines.

### _lwork wrapper issues

- Multiple fixes for character args (null char->proper bytes), array args typed as scalars, work/rwork type from routine prefix, scalar/array confusion with `np.PyArray_DATA`.

### Callback (gees/gges)

- Hand-written 8 routines with `cdef` callback functions using `noexcept nogil` + `with gil`. Uses `_get_max_nargs` with `inspect.signature` and try/except fallback for arg count compatibility.

### check() validation

- Translated f2py check() directives to Python validation. Error class extends ValueError. Dimension shape checks for 1D (simple expressions only) and 2D arrays. Hidden arg checks included (e.g., trmm's `k` validation).

### Post-call array copy

- `for(i=0;i<N;i++){dst[i]=src[i];}` regex needed optional trailing `}` for consecutive loops.

### Read-only pivot arrays

- `piv -= 1` on read-only arrays from `broadcast_to`. Added writability check.

## Problem Solving

### Solved

- Complete f2py replacement with Cython wrappers (6471 per-file tests pass, 0 failures)
- All segfaults from pre-call pivot issues (getrs, getri, gbcon, gbtrs, laswp, gesc2)
- integrate test_integration crash (Radau/BDF using lu_solve with wrong pivots)
- lobpcg test crash (same pivot issue)
- Schur decomposition callbacks (gees/gges with 8 hand-written routines)
- Workspace queries (_lwork functions for 116 routines)
- Batch test failures (cho_solve lower array, overwrite params, read-only pivots)
- Input validation (check() directives, dimension shape checks)
- f2py infrastructure removal (build system, tools, ILP64 files)
- COS-sin decomposition (154 tests fixed via _lwork rwork/iwork returns)
- Python 3 bytes indexing in dimension expressions (20 occurrences)
- gejsv iworkout post-call copy and edge arguments (0D scalar input)
- gtsvx validation (dimension split, 1D checks, None guard, ValueError base class)

### Ongoing

- Cross-test memory corruption when running thousands of tests in one process (segfaults in test_lapack.py full file, test_gttrf_gttrs via pytest). Individual tests pass. Needs ASan build.
- ILP64 variant not yet implemented

## User Messages (Chronological)

1. "Please review the remaining usages of f2py (should be only in scipy/linalg/) and determine options to get rid of that dependency. Can we use the Cython BLAS and LAPACK wrappers instead?"
2. "For phase 5, wouldn't it be better to change _pyblas.pyx and _pylapack.pyx to use blas_int and rename them to .pyx.tmpl so we can keep the source common and generate the _lp64 and _ilp64 variants from it? Or are there more differences than integer types?"
3. "Please write out this last plan to a file f2py_removal_plan.md in the root of the repo. Include all alternatives A/B/C/D"
4. [Approved plan]
5. "please commit phase 1, then continue"
6. "use `pixi run python` instead of `python`. Then, execute"
7. "do not `cd` one level up, stay in the root of this repo"
8. "if you want to use numpy or scipy, please use `pixi run -e test spin python --no-build -- -c` instead of `pixi run python -c`"
9. "please continue" (multiple times)
10. "Can you fix the 8 gees/gges callbacks and TestSchur failures?"
11. "please continue on test_batch"
12. "`lower` and `trans` seem inherently scalar, so there probably isn't a need to convert them to arrays? Would you agree that the best fix is to modify the batch framework?"
13. "go ahead with (1)" [removing f2py infrastructure]
14. "Can you explain why you think `stbsv` isn't covered in scipy/linalg/fblas_l2.pyf.src?"
15. "Why do you think we need those 4 missing utility BLAS functions? There's in cython_blas but not in the f2py wrappers, so are they actually missing?"
16. "`pixi r test -t scipy.linalg.tests.test_blas` crashes immediately, can you verify and fix?"
17. "This shows a segfault as well: `pixi r test -t scipy.sparse.linalg._eigen.lobpcg.tests.test_lobpcg`"
18. "There are more crashes in `optimize` and `sparse.linalg` submodules. And there's now one in `test_decomp`. How would you like to approach this?"
19. "fix CHEEVD first, then see what is left. If there are still multiple issues at that point, switch to the systematic approach"
20. "test_batch now segfaults."
21. "For (2), please show me the signatures of both a representative function, and of the underlying BLAS library"
22. "For (3), please implement the check() validation so the tests pass"
23. "This shows a segfault: `pixi r test -s linalg -v -- -k gejsv`"
24. "It's a single-test crash: `pixi r test -s linalg -v -- -k gejsv_edge_arguments`"
25. "Next failures: `pixi r test -s linalg -v -- -k gtsvx`"
26. "Next segfault: `pixi r test -s linalg -v -- -k test_gttrf_gttrs`"
27. "Your conclusion seems wrong, this reliably segfaults for me: `pixi r test -s linalg -v -- -k test_gttrf_gttrs`"
28. "This may be good to debug next: `pixi r test -s linalg -v -- -k gejsv`" [separate from earlier gejsv]
29. "What is left now, only implementing ILP64 support?"

## User Feedback Corrections

- Use `pixi run python` not bare `python`
- Use `pixi run -e test spin python --no-build` for scipy imports
- `stbsv` IS in the pyf file (parser was losing it due to end-subroutine regex bug)
- `lower`/`trans` are scalar flags, best fix is in batch framework (leading to cho_factor fix)
- 4 utility BLAS functions aren't actually missing (never in f2py either)
- An empty commit was created (only a blank line diff) - pointed out by user
- Conclusion about gttrf_gttrs was wrong - it reliably segfaults

## Pending Tasks

- **ILP64 variant**: Generate `_pyblas_64`/`_pylapack_64` with `blas_int=int64_t`
- **Cross-test memory corruption**: test_gttrf_gttrs segfaults via pytest but passes in direct Python. Needs ASan instrumented build to find the source. The gttrs wrapper itself is correct.
- **`_cython_signature_generator.py`**: Still imports `numpy.f2py.crackfortran` (developer-only tool, not a build dependency)

## Current State

The most recent work was debugging the `test_gttrf_gttrs` segfault reported by the user with `pixi r test -s linalg -v -- -k test_gttrf_gttrs`. Investigation revealed:

- The test passes in direct Python invocation but crashes through pytest
- An attempted fix to reorder args (putting `b` before `ipiv` in sgttrs) was WRONG - the test expects ipiv before b (matching Fortran/pyf arg_names order)
- The arg reordering was fully reverted
- An empty commit (just a blank line diff) was created and then removed after user pointed it out
- Current state: 45 commits on f2py-removal branch, clean working tree
- The segfault remains unresolved - appears to be latent memory corruption from another wrapper that only manifests when pytest loads many extension modules

The last clean commit is `071b455fbb ENH: linalg: fix gtsvx validation - dimension split, 1D checks, error class`.
