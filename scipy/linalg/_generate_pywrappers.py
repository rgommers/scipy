#!/usr/bin/env python3
"""
Code generator that produces Cython `def` wrappers around the existing
cython_blas/cython_lapack `cdef` functions.

These generated wrappers replace the f2py-generated _fblas/_flapack modules
by providing equivalent Python-callable interfaces with:
- NumPy array input/output
- Automatic Fortran-order conversion and dtype casting
- Dimension inference from array shapes
- Input validation (checks)
- Copy/overwrite semantics
- Workspace allocation for hidden work arrays

Usage::

    python _generate_pywrappers.py -o <outdir> [--ilp64]

This reads the .pyf.src files and cython_*_signatures.txt to generate:
- _pyblas.pyx / _pyblas.pxd  (or _pyblas_64 for ILP64)
- _pylapack.pyx / _pylapack.pxd  (or _pylapack_64 for ILP64)
"""

import argparse
import os
import re
import textwrap

# We use importlib to avoid triggering scipy's __init__.py
import importlib.util


def _import_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Import _extract_balanced_parens from _pyf_parser without triggering scipy import
_pyf_parser = _import_module_from_file(
    '_pyf_parser', os.path.join(BASE_DIR, '_pyf_parser.py')
)
_extract_balanced_parens = _pyf_parser._extract_balanced_parens


def _load_cdef_signatures(signature_file):
    """Load cdef function signatures from a cython_*_signatures.txt file.

    Returns a dict mapping function name to list of parameter type strings.
    E.g., {'crotg': ['c', 'c', 's', 'c'], 'dgemm': ['char', 'char', ...]}
    """
    sigs = {}
    with open(os.path.join(BASE_DIR, signature_file)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: 'void caxpy(int *n, c *ca, ...)' or 'float sdot(...)'
            paren_idx = line.find('(')
            if paren_idx == -1:
                continue
            header = line[:paren_idx].strip().split()
            if len(header) < 2:
                continue
            name = header[-1]
            # Parse parameter list
            params_str = line[paren_idx+1:line.rfind(')')]
            param_types = []
            for param in params_str.split(','):
                param = param.strip()
                if not param:
                    continue
                # "type *name" or "type name" - extract type
                parts = param.replace('*', ' * ').split()
                # Type is everything before the last token (the name)
                ptype = ' '.join(parts[:-1]).replace(' * ', ' *').replace('* ', '*').strip()
                # Normalize: remove trailing *
                ptype = ptype.rstrip('*').strip()
                param_types.append(ptype)
            sigs[name] = param_types
    return sigs


def _load_cdef_names(signature_file):
    """Load available cdef function names from a cython_*_signatures.txt file."""
    return set(_load_cdef_signatures(signature_file).keys())


# Map f2py Fortran types to numpy dtype strings and C types
FTYPE_TO_NUMPY = {
    'real': 'np.float32',
    'double precision': 'np.float64',
    'complex': 'np.complex64',
    'double complex': 'np.complex128',
    'integer': 'np.intc',  # will be overridden for blas_int
    'logical': 'np.intc',  # Fortran LOGICAL mapped to C int
}

FTYPE_TO_CTYPE = {
    'real': 'float',
    'double precision': 'double',
    'complex': 'float complex',
    'double complex': 'double complex',
}

# cython_blas/cython_lapack use short typedefs: s, d, c, z
# We import these with cy_ prefix to avoid name collisions (e.g., with
# a variable named 'c' in crotg). These are used for pointer casts.
FTYPE_TO_CYTYPE = {
    'real': 'cy_s',
    'double precision': 'cy_d',
    'complex': 'cy_c',
    'double complex': 'cy_z',
}

# Map ftype to single-char prefix
FTYPE_TO_PREFIX = {
    'real': 's',
    'double precision': 'd',
    'complex': 'c',
    'double complex': 'z',
}

def _is_simple_literal(s):
    """Check if a string is a simple compile-time literal (int, float, complex)."""
    s = s.strip()
    # Integer
    if re.match(r'^-?\d+$', s):
        return True
    # Float
    if re.match(r'^-?\d+\.?\d*$', s):
        return True
    # Complex tuple like (0.0,0.0) -> convert to Python complex
    if re.match(r'^\([\d.,\s-]+\)$', s):
        return True
    return False


def _format_literal(s, ftype):
    """Format a literal value for Cython, handling complex numbers."""
    s = s.strip()
    # Complex tuple like (0.0,0.0) or (1.0\,0.0) -> Python complex literal
    m = re.match(r'^\((-?[\d.]+)\s*,\s*(-?[\d.]+)\)$', s)
    if m:
        real = float(m.group(1))
        imag = float(m.group(2))
        if imag == 0:
            return str(real)
        elif real == 0:
            return f'{imag}j'
        else:
            return f'({real}+{imag}j)'
    return s


COMMENT_HEADER = """\
# This file was generated by _generate_pywrappers.py.
# Do not edit this file directly.
"""


def _get_python_args(routine):
    """Get the list of Python-visible arguments for a routine.

    Hidden args (intent(hide)) and cache args are excluded.
    Arguments are returned in a sensible order: required args first,
    then optional args. Required arrays, then required scalars, then
    optional keyword args.
    """
    required = []
    optional = []
    for aname in routine['arg_names']:
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        if 'hide' in intents:
            continue
        if 'cache' in intents:
            continue

        # Pure output scalars (intent(out) without intent(in)) are not
        # passed by the user - they're allocated internally and returned.
        # But output ARRAYS may be optionally passed for in-place operation.
        if 'out' in intents and 'in' not in intents and not _is_array_arg(ainfo):
            continue

        is_opt = ainfo.get('optional', False)
        has_default = 'default' in ainfo
        # Output-only arrays are optional (user can let us allocate)
        if 'out' in intents and 'in' not in intents and _is_array_arg(ainfo):
            optional.append(aname)
        elif is_opt or has_default:
            optional.append(aname)
        else:
            required.append(aname)

    # Also add any args that are not in arg_names but have 'out' intent
    for aname, ainfo in routine['args'].items():
        if aname not in routine['arg_names']:
            intents = ainfo.get('intents', [])
            if 'out' in intents and aname not in required and aname not in optional:
                # Pure output args that aren't in the signature are
                # typically allocated internally, not passed by user
                pass

    return required + optional


def _get_hidden_args(routine):
    """Get args with intent(hide), topologically sorted by dependencies.

    Hidden args often depend on each other (e.g., m depends on lda which
    depends on shape(a,0)). We sort them so that each arg is computed
    after its dependencies.
    """
    hidden = []
    for aname in routine['arg_names']:
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        if 'hide' in intents or 'cache' in intents:
            hidden.append(aname)

    # Topological sort based on depend() chains
    # Only consider dependencies between hidden args
    hidden_set = set(hidden)
    sorted_hidden = []
    visited = set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        ainfo = routine['args'].get(name, {})
        for dep in ainfo.get('depend', []):
            if dep in hidden_set:
                visit(dep)
        sorted_hidden.append(name)

    for name in hidden:
        visit(name)

    return sorted_hidden


def _get_output_args(routine):
    """Get args that should be returned (intent out or in,out)."""
    outputs = []
    for aname, ainfo in routine['args'].items():
        intents = ainfo.get('intents', [])
        if 'out' in intents:
            outputs.append(aname)
    return outputs


def _is_array_arg(ainfo):
    """Check if an argument is an array (has dimension)."""
    return ainfo.get('dimension') is not None


def _translate_f2py_expr(expr, routine_args):
    """Translate an f2py expression to Python.

    Handles common patterns like:
    - shape(a, 0) -> a.shape[0]
    - len(x) -> x.shape[0]  (for 1D arrays)
    - MAX(a, b) -> max(a, b)
    - C ternary: (cond ? val1 : val2) -> (val1 if cond else val2)
    """
    if expr is None:
        return None

    result = expr.strip()

    # shape(var, idx) -> var.shape[idx]
    result = re.sub(r'shape\((\w+)\s*,\s*(\d+)\)', r'\1.shape[\2]', result)

    # len(var) -> var.shape[0]
    result = re.sub(r'len\((\w+)\)', r'\1.shape[0]', result)

    # MAX(a, b) -> max(a, b) (case insensitive)
    result = re.sub(r'MAX\(', 'max(', result)
    result = re.sub(r'MIN\(', 'min(', result)

    # abs() is already Python-compatible

    # f2py internal variables: var_capi==Py_None -> var is None
    result = re.sub(r'(\w+)_capi==Py_None', r'\1 is None', result)
    result = re.sub(r'(\w+)_capi!=Py_None', r'\1 is not None', result)

    # C pointer dereference for char comparisons: *var=='X' -> var == ord('X')
    # These appear in dimension expressions where char variables are compared
    # The char variables are actually integer indices in the Python wrappers
    def _deref_char_cmp(m):
        var = m.group(1)
        char = m.group(2)
        # Map common LAPACK character options to their integer indices
        # These match the string lookup tables used in the callstatements
        _char_maps = {
            # range: A=0, V=1, I=2
            'A': 0, 'V': 1, 'I': 2,
            # job: N=0, V=1, etc.
            'N': 0, 'T': 1, 'C': 2,
        }
        idx = _char_maps.get(char, f"ord('{char}')")
        return f'{var} == {idx}'

    result = re.sub(r'\*(\w+)==[\'"](.)[\'"]', _deref_char_cmp, result)

    # C ternary expressions: (cond ? a : b) -> (a if cond else b)
    # Handle min/max patterns: (a <= b ? a : b) -> min(a, b)
    result = _translate_min_max_ternary(result)
    # Then generic ternaries
    result = _translate_ternary(result)

    # C logical operators
    result = result.replace('&&', ' and ')
    result = result.replace('||', ' or ')

    return result


def _translate_min_max_ternary(expr):
    """Translate C min/max ternary patterns to Python min()/max().

    (a <= b ? a : b) -> min(a, b)
    (a >= b ? a : b) -> max(a, b)
    """
    # We need to handle this with balanced paren matching since the
    # expressions can be complex. Look for the outermost ternary.
    # Pattern: (EXPR_A OP EXPR_B ? EXPR_C : EXPR_D)
    # where EXPR_C ~= EXPR_A and EXPR_D ~= EXPR_B
    if '?' not in expr:
        return expr

    # Try to find (a <= b ? a : b) or (a >= b ? a : b) pattern
    # Use a simpler approach: find ? and : at the same paren depth
    for attempt in range(5):  # max iterations
        q_idx = expr.find('?')
        if q_idx == -1:
            break

        # Find the matching : at the same depth
        depth = 0
        colon_idx = -1
        for i in range(q_idx + 1, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth < 0:
                    break
            elif expr[i] == ':' and depth == 0:
                colon_idx = i
                break
            elif expr[i] == '?' and depth == 0:
                break  # nested ternary, skip

        if colon_idx == -1:
            break

        # Find the opening paren before the condition
        # Walk backwards from ? to find <=, >=
        cond_str = expr[:q_idx].rstrip()
        if '<=' in cond_str:
            op_idx = cond_str.rfind('<=')
            a_part = cond_str[:op_idx].rstrip()
            b_part = cond_str[op_idx+2:].lstrip()
            val_true = expr[q_idx+1:colon_idx].strip()
            val_false = expr[colon_idx+1:].strip()

            # Strip outer parens
            if a_part.startswith('('):
                a_part = a_part[1:]
            if val_false.endswith(')'):
                val_false = val_false[:-1]

            # Check if it's a min pattern: (a <= b ? a : b)
            if (a_part.strip().replace(' ', '') ==
                    val_true.strip().replace(' ', '') and
                b_part.strip().replace(' ', '') ==
                    val_false.strip().replace(' ', '')):
                return f'min({val_true.strip()}, {val_false.strip()})'

        if '>=' in cond_str:
            op_idx = cond_str.rfind('>=')
            a_part = cond_str[:op_idx].rstrip()
            b_part = cond_str[op_idx+2:].lstrip()
            val_true = expr[q_idx+1:colon_idx].strip()
            val_false = expr[colon_idx+1:].strip()

            if a_part.startswith('('):
                a_part = a_part[1:]
            if val_false.endswith(')'):
                val_false = val_false[:-1]

            if (a_part.strip().replace(' ', '') ==
                    val_true.strip().replace(' ', '') and
                b_part.strip().replace(' ', '') ==
                    val_false.strip().replace(' ', '')):
                return f'max({val_true.strip()}, {val_false.strip()})'

        break  # Don't loop if not a min/max pattern

    return expr


def _translate_ternary(expr):
    """Translate C ternary (cond ? a : b) to Python (a if cond else b).

    Handles nested parens like ((compute_v==1)?n:1) and nested ternaries.
    """
    if '?' not in expr:
        return expr

    # Strip outer balanced parens first
    stripped = _strip_outer_parens(expr)
    if stripped != expr:
        result = _translate_ternary(stripped)
        if result != stripped:
            return f'({result})'

    # Find ? at depth 0
    depth = 0
    q_idx = -1
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '?' and depth == 0:
            q_idx = i
            break

    if q_idx == -1:
        # No ternary at depth 0 - recurse into parenthesized subexpressions
        if '?' in expr:
            return _translate_ternary_recursive(expr)
        return expr

    # Find : at depth 0, after ?
    depth = 0
    colon_idx = -1
    for i in range(q_idx + 1, len(expr)):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
        elif expr[i] == ':' and depth == 0:
            colon_idx = i
            break

    if colon_idx == -1:
        return expr

    cond = expr[:q_idx].strip()
    val_true = expr[q_idx+1:colon_idx].strip()
    val_false = expr[colon_idx+1:].strip()

    # Recursively translate nested ternaries
    val_true = _translate_ternary(val_true)
    val_false = _translate_ternary(val_false)

    return f'({val_true} if {cond} else {val_false})'


def _translate_ternary_recursive(expr):
    """Translate ternaries inside parenthesized subexpressions."""
    result = []
    i = 0
    while i < len(expr):
        if expr[i] == '(':
            # Find matching close paren
            content, end = _extract_balanced_parens(expr, i)
            # Recursively translate the content
            translated = _translate_ternary(content)
            result.append(f'({translated})')
            i = end
        else:
            result.append(expr[i])
            i += 1
    return ''.join(result)


def _strip_outer_parens(s):
    """Strip one layer of balanced outer parentheses if present."""
    s = s.strip()
    if not s.startswith('(') or not s.endswith(')'):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                return s  # closing paren isn't the last char
    if depth == 0:
        return s[1:-1]
    return s


def _get_numpy_dtype(ftype):
    """Get numpy dtype for a Fortran type."""
    return FTYPE_TO_NUMPY.get(ftype, 'np.float64')


def _get_overwrite_param_name(aname):
    """Get the overwrite_X parameter name for a copy intent arg."""
    return f'overwrite_{aname}'


def _generate_wrapper_function(routine, lib_module_name, cdef_param_types=None):
    """Generate a Cython def wrapper function for one routine.

    Parameters
    ----------
    routine : dict
        Parsed routine specification from _pyf_parser.
    lib_module_name : str
        'cython_blas' or 'cython_lapack'
    cdef_param_types : list of str, optional
        Parameter types from cython_*_signatures.txt (e.g., ['c', 'c', 's', 'c']).
        Used for correct pointer casts when calling the cdef function.

    Returns
    -------
    code : str
        Cython source code for the wrapper function.
    """
    name = routine['name']
    is_function = routine['type'] == 'function'
    py_args = _get_python_args(routine)
    hidden_args = _get_hidden_args(routine)
    output_args = _get_output_args(routine)

    # Determine the primary data type from first array arg or first typed arg
    primary_ftype = None
    for aname in routine['arg_names']:
        ainfo = routine['args'].get(aname, {})
        ft = ainfo.get('ftype')
        if ft and ft != 'integer':
            primary_ftype = ft
            break
    if primary_ftype is None:
        primary_ftype = 'double precision'

    numpy_dtype = _get_numpy_dtype(primary_ftype)

    lines = []

    # Build function signature
    # Defaults that reference other args can't be in the signature -
    # they must be computed in the function body using sentinel values.
    sig_parts = []
    body_defaults = []  # (argname, expr) for defaults computed in body

    for aname in py_args:
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        default = ainfo.get('default')
        is_optional = ainfo.get('optional', False)

        if _is_array_arg(ainfo):
            # Array args: accept as object, convert later
            if is_optional or ('out' in intents and 'in' not in intents):
                sig_parts.append(f'{aname}=None')
            else:
                sig_parts.append(aname)
        elif ainfo.get('ftype') == 'character':
            # Character args: accept as str or bytes, convert to bytes
            if default is not None:
                char_default = default.replace('"', '').replace("'", '')
                sig_parts.append(f'{aname}=b"{char_default}"')
            else:
                sig_parts.append(aname)
        elif ainfo.get('ftype') in ('integer', 'logical'):
            if (is_optional or default is not None) and default is not None:
                if _is_simple_literal(default):
                    sig_parts.append(f'int {aname}={default}')
                else:
                    sig_parts.append(f'int {aname}=-1')
                    py_expr = _translate_f2py_expr(default, routine['args'])
                    body_defaults.append((aname, py_expr))
            else:
                sig_parts.append(f'int {aname}')
        else:
            # Scalar of the routine's type - must be typed for Cython
            ftype = ainfo.get('ftype', primary_ftype)
            ctype = FTYPE_TO_CTYPE.get(ftype, 'double')
            if (is_optional or default is not None) and default is not None:
                if _is_simple_literal(default):
                    formatted = _format_literal(default, ftype)
                    sig_parts.append(f'{ctype} {aname}={formatted}')
                else:
                    sig_parts.append(f'{ctype} {aname}=0')
                    py_expr = _translate_f2py_expr(default, routine['args'])
                    body_defaults.append((aname, py_expr))
            else:
                sig_parts.append(f'{ctype} {aname}')

    # Add overwrite_ parameters for arrays with copy, overwrite, or in,out intent
    for aname in py_args:
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        if not _is_array_arg(ainfo):
            continue
        if ('copy' in intents or 'overwrite' in intents) and 'in' in intents:
            # intent(in,copy), intent(in,overwrite), or intent(in,out,copy)
            ow_name = _get_overwrite_param_name(aname)
            sig_parts.append(f'int {ow_name}=0')
        elif 'in' in intents and 'out' in intents:
            # in,out without copy
            ow_name = _get_overwrite_param_name(aname)
            sig_parts.append(f'int {ow_name}=0')

    sig = ', '.join(sig_parts)
    lines.append(f'def {name}({sig}):')

    # Docstring (minimal)
    lines.append(f'    """Wrapper for ``{name}``."""')

    # --- Variable declarations ---
    # Collect all cdef declarations first, then emit block only if non-empty
    cdef_lines = []

    # Declare blas_int variables for hidden integer SCALAR args
    # (skip integer arrays like iwork which have dimension)
    int_vars = []
    for aname in hidden_args:
        ainfo = routine['args'].get(aname, {})
        if ainfo.get('ftype') in ('integer', 'logical') and not ainfo.get('dimension'):
            int_vars.append(aname)
    if int_vars:
        cdef_lines.append(f'        blas_int {", ".join(int_vars)}')

    # Declare output scalar variables (intent(out) without intent(in))
    for aname in routine['arg_names']:
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        if 'out' in intents and 'in' not in intents and not _is_array_arg(ainfo):
            ftype = ainfo.get('ftype', 'integer')
            if ftype == 'integer':
                cdef_lines.append(f'        blas_int {aname}')
            else:
                ctype = FTYPE_TO_CTYPE.get(ftype, 'double')
                cdef_lines.append(f'        {ctype} {aname}')

    # Declare function return variable if needed
    if is_function and routine.get('result_name'):
        rname = routine['result_name']
        rtype = routine.get('return_type')
        if not rtype:
            for vname in [rname, routine['name']]:
                if vname in routine['args'] and 'ftype' in routine['args'][vname]:
                    rtype = routine['args'][vname]['ftype']
                    break
        if not rtype:
            rtype = primary_ftype
        ctype = FTYPE_TO_CTYPE.get(rtype, 'double')
        cdef_lines.append(f'        {ctype} {rname}')

    if cdef_lines:
        lines.append('    cdef:')
        lines.extend(cdef_lines)

    # NOTE: body_defaults are emitted AFTER hidden args, not here.
    # Many computed defaults depend on hidden args like n, m, etc.

    # --- Input validation and array conversion ---
    lines.append('')

    # Ensure arrays have the expected dimensionality (f2py does this
    # automatically, but our wrappers need to be explicit).
    # A 1D array passed to a 2D parameter gets reshaped to (n, 1),
    # and the output is squeezed back to 1D on return.
    _reshaped_args = []
    for aname in py_args:
        ainfo = routine['args'].get(aname, {})
        dim = ainfo.get('dimension', '')
        if not dim or dim == '*':
            continue
        ndim = len([d for d in dim.split(',') if d.strip()])
        if ndim >= 2:
            lines.append(f'    _was_1d_{aname} = ({aname} is not None) and np.ndim({aname}) == 1')
            lines.append(f'    if _was_1d_{aname}:')
            lines.append(f'        {aname} = np.asarray({aname}).reshape(-1, 1)')
            _reshaped_args.append(aname)

    # Convert character args from str to bytes if needed
    for aname in py_args:
        ainfo = routine['args'].get(aname, {})
        if ainfo.get('ftype') == 'character':
            lines.append(f'    if isinstance({aname}, str):')
            lines.append(f'        {aname} = {aname}.encode()')

    # Process array arguments: convert to Fortran-contiguous, correct dtype
    for aname in py_args:
        ainfo = routine['args'].get(aname, {})
        if not _is_array_arg(ainfo):
            continue

        intents = ainfo.get('intents', [])
        ftype = ainfo.get('ftype', primary_ftype)
        dt = _get_numpy_dtype(ftype)
        is_optional = ainfo.get('optional', False)

        if 'copy' in intents and 'in' in intents and 'out' in intents:
            # intent(in,out,copy) - copy unless overwrite
            ow_name = _get_overwrite_param_name(aname)
            if is_optional:
                lines.append(f'    if {aname} is not None:')
                lines.append(f'        if not {ow_name}:')
                lines.append(f'            {aname} = np.array({aname}, dtype={dt}, order="F", copy=True)')
                lines.append(f'        else:')
                lines.append(f'            {aname} = np.asfortranarray({aname}, dtype={dt})')
            else:
                lines.append(f'    if not {ow_name}:')
                lines.append(f'        {aname} = np.array({aname}, dtype={dt}, order="F", copy=True)')
                lines.append(f'    else:')
                lines.append(f'        {aname} = np.asfortranarray({aname}, dtype={dt})')
        elif 'in' in intents and 'out' in intents:
            # intent(in,out) without copy - still handle overwrite
            ow_name = _get_overwrite_param_name(aname)
            lines.append(f'    if not {ow_name}:')
            lines.append(f'        {aname} = np.array({aname}, dtype={dt}, order="F", copy=True)')
            lines.append(f'    else:')
            lines.append(f'        {aname} = np.asfortranarray({aname}, dtype={dt})')
        elif 'in' in intents and ('copy' in intents or 'overwrite' in intents):
            # intent(in,copy) or intent(in,overwrite) - copy unless overwrite
            ow_name = _get_overwrite_param_name(aname)
            lines.append(f'    if not {ow_name}:')
            lines.append(f'        {aname} = np.array({aname}, dtype={dt}, order="F", copy=True)')
            lines.append(f'    else:')
            lines.append(f'        {aname} = np.asfortranarray({aname}, dtype={dt})')
        elif 'in' in intents:
            # intent(in) only - just ensure correct dtype and order
            lines.append(f'    {aname} = np.asfortranarray({aname}, dtype={dt})')
        elif 'out' in intents:
            # intent(out) only - will be allocated below
            pass

    # --- Compute hidden args ---
    # Some hidden args reference .shape of output arrays that haven't
    # been allocated yet. We compute these after output allocation.
    _output_only_names = set()
    for _aname in routine['arg_names']:
        _ainfo = routine['args'].get(_aname, {})
        _intents = _ainfo.get('intents', [])
        if 'out' in _intents and 'in' not in _intents and _is_array_arg(_ainfo):
            _output_only_names.add(_aname)

    lines.append('')
    _deferred_hidden = []
    for aname in hidden_args:
        ainfo = routine['args'].get(aname, {})
        default = ainfo.get('default')
        if default is None:
            continue
        # Check if default expression references shape of a pure output array
        default_lower = default.lower()
        needs_defer = False
        for out_name in _output_only_names:
            if f'shape({out_name}' in default_lower or f'{out_name}.shape' in default_lower:
                needs_defer = True
                break
        if needs_defer:
            _deferred_hidden.append(aname)
        else:
            py_expr = _translate_f2py_expr(default, routine['args'])
            lines.append(f'    {aname} = {py_expr}')

    # --- Computed defaults (placed after hidden args are computed) ---
    if body_defaults:
        lines.append('')
        for aname, expr in body_defaults:
            ainfo = routine['args'].get(aname, {})
            if ainfo.get('ftype') in ('integer', 'logical'):
                lines.append(f'    if {aname} == -1:')
                lines.append(f'        {aname} = {expr}')
            else:
                lines.append(f'    if {aname} is None:')
                lines.append(f'        {aname} = {expr}')

    # --- Allocate output-only arrays ---
    lines.append('')
    for aname in py_args:
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        dim = ainfo.get('dimension')
        ftype = ainfo.get('ftype', primary_ftype)
        dt = _get_numpy_dtype(ftype)

        if 'out' in intents and 'in' not in intents and dim:
            # Pure output array - allocate
            shape = _translate_dimension_to_shape(dim)
            lines.append(f'    {aname} = np.empty({shape}, dtype={dt}, order="F")')
        elif 'out' in intents and 'in' in intents and dim:
            is_optional = ainfo.get('optional', False)
            if is_optional:
                # Optional in,out array - allocate if not provided
                shape = _translate_dimension_to_shape(dim)
                lines.append(f'    if {aname} is None:')
                lines.append(f'        {aname} = np.zeros({shape}, dtype={dt}, order="F")')

    # --- Compute deferred hidden args (those that reference output array shapes) ---
    for aname in _deferred_hidden:
        ainfo = routine['args'].get(aname, {})
        default = ainfo.get('default')
        if default is not None:
            py_expr = _translate_f2py_expr(default, routine['args'])
            lines.append(f'    {aname} = {py_expr}')

    # --- Allocate hidden workspace arrays ---
    for aname in hidden_args:
        ainfo = routine['args'].get(aname, {})
        dim = ainfo.get('dimension')
        ftype = ainfo.get('ftype')
        if dim and ftype and ftype != 'integer':
            dt = _get_numpy_dtype(ftype)
            shape = _translate_dimension_to_shape(dim)
            lines.append(f'    {aname} = np.empty({shape}, dtype={dt}, order="F")')
        elif dim and ftype == 'integer':
            shape = _translate_dimension_to_shape(dim)
            lines.append(f'    {aname} = np.empty({shape}, dtype=np.intc, order="F")')

    # --- Pre-call variable initializations (from compound callstatements) ---
    cs = routine.get('callstatement')
    if cs and cs.startswith('{'):
        parsed_cs = _parse_callstatement(routine)
        if parsed_cs and parsed_cs['pre_call']:
            # Rejoin the pre_call parts to reconstruct for loops
            # (they get split by ';' which appears inside for(...;...;...))
            raw_pre = ';'.join(parsed_cs['pre_call'])
            # Process each statement
            for stmt in parsed_cs['pre_call']:
                stmt = stmt.strip()
                # Handle: F_INT i=expr or F_INT i
                m = re.match(r'F_INT\s+(\w+)\s*=\s*(.*)', stmt)
                if m:
                    var = m.group(1)
                    expr = _translate_f2py_expr(m.group(2), routine['args'])
                    lines.append(f'    cdef blas_int {var} = {expr}')
                elif re.match(r'F_INT\s+(\w+)', stmt):
                    var = re.match(r'F_INT\s+(\w+)', stmt).group(1)
                    lines.append(f'    cdef blas_int {var}')
                # Handle: var++ or var--
                elif re.match(r'(\w+)\+\+$', stmt):
                    var = re.match(r'(\w+)\+\+$', stmt).group(1)
                    lines.append(f'    {var} += 1')
                elif re.match(r'(\w+)--$', stmt):
                    var = re.match(r'(\w+)--$', stmt).group(1)
                    lines.append(f'    {var} -= 1')

            # Handle for loops that increment/decrement arrays
            # Patterns in the rejoined pre-call string:
            #   for(i=0;i<N;++arr[i++])
            #   for(i=0;i<n;++ipiv[i],++jpiv[i++])
            for m in re.finditer(r'\+\+(\w+)\[', raw_pre):
                arr = m.group(1)
                if arr == 'i':  # skip loop variable
                    continue
                lines.append(f'    {arr} = np.array({arr}, copy=True)')
                lines.append(f'    {arr} += 1  # Convert 0-based to 1-based')
            for m in re.finditer(r'--(\w+)\[', raw_pre):
                arr = m.group(1)
                if arr == 'i':
                    continue
                lines.append(f'    {arr} = np.array({arr}, copy=True)')
                lines.append(f'    {arr} -= 1  # Convert 1-based to 0-based')

    # --- Call the low-level cdef function ---
    lines.append('')
    call_args = _build_call_args(routine, lib_module_name, cdef_param_types)

    # Check for wrapped complex return functions (cdotu, cdotc, zdotu, zdotc)
    wrp_return_var = routine.pop('_wrp_return_var', None)
    if wrp_return_var:
        lines.append(f'    {wrp_return_var} = {lib_module_name}.{name}({call_args})')
    elif is_function and routine.get('result_name'):
        result_var = routine['result_name']
        lines.append(f'    {result_var} = {lib_module_name}.{name}({call_args})')
    else:
        lines.append(f'    {lib_module_name}.{name}({call_args})')

    # --- Post-call processing ---
    post_call = _generate_post_call(routine)
    if post_call:
        lines.append('')
        lines.extend(post_call)

    # --- Build return value ---
    lines.append('')
    return_parts = _build_return(routine)
    if return_parts:
        # Squeeze back arrays that were reshaped from 1D to 2D
        for rp in return_parts:
            var = rp['var']
            if var in _reshaped_args:
                lines.append(f'    if _was_1d_{var}:')
                lines.append(f'        {var} = {var}.reshape(-1)')
        return_vars = [r['var'] for r in return_parts]
        lines.append(f'    return {", ".join(return_vars)}')

    return '\n'.join(lines) + '\n'


def _translate_dimension_to_shape(dim):
    """Translate an f2py dimension string to a Python shape tuple.

    '(m,n)' -> '(m, n)'
    '*' -> '(n,)'  (handled separately)
    'n' -> '(n,)'
    'lda, n' -> '(lda, n)'
    """
    dim = dim.strip()
    if dim == '*':
        return None  # 1D, size determined from input

    # Translate f2py expressions
    dim = _translate_f2py_expr(dim, {})

    parts = [p.strip() for p in dim.split(',')]
    if len(parts) == 1:
        return f'({parts[0]},)'
    else:
        return f'({", ".join(parts)})'


def _build_call_args(routine, lib_module_name, cdef_param_types=None):
    """Build the argument string for calling the cython_blas/lapack cdef function."""
    cs = routine.get('callstatement')
    if cs:
        return _translate_callstatement_args(routine, cdef_param_types)

    # No callstatement - build from arg_names.
    # Use cdef_param_types (from cython_*_signatures.txt) for correct
    # pointer casts, since the pyf callprotoargument may have wrong types.
    _sig_to_cytype = {
        's': 'cy_s', 'd': 'cy_d', 'c': 'cy_c', 'z': 'cy_z',
        'char': None,
        'int': 'blas_int', 'blas_int': 'blas_int', 'bint': 'blas_int',
    }

    args = []
    for i, aname in enumerate(routine['arg_names']):
        ainfo = routine['args'].get(aname, {})
        # Get the expected type from the cdef signature
        sig_type = cdef_param_types[i] if (cdef_param_types and
                                           i < len(cdef_param_types)) else None
        cytype = _sig_to_cytype.get(sig_type) if sig_type else None

        if _is_array_arg(ainfo):
            if cytype:
                args.append(f'<{cytype} *>np.PyArray_DATA({aname})')
            else:
                args.append(f'<{_ctype_ptr(ainfo)}>np.PyArray_DATA({aname})')
        else:
            if cytype:
                args.append(f'<{cytype} *>&{aname}')
            else:
                args.append(f'&{aname}')
    return ', '.join(args)


def _ctype_ptr(ainfo):
    """Get Cython pointer type for an arg.

    Uses aliased cython_blas typedefs (cy_s, cy_d, cy_c, cy_z) for
    compatibility with cython_blas/cython_lapack cdef function signatures.
    """
    ftype = ainfo.get('ftype', 'double precision')
    if ftype == 'integer':
        return 'blas_int *'
    cytype = FTYPE_TO_CYTYPE.get(ftype)
    if cytype:
        return f'{cytype} *'
    ctype = FTYPE_TO_CTYPE.get(ftype, 'double')
    return f'{ctype} *'


def _parse_callstatement(routine):
    """Parse the callstatement into structured components.

    Returns a dict with:
    - 'pre_call': list of C statements before the function call
    - 'call_args': list of raw argument expressions
    - 'post_call': list of C statements after the function call
    - 'return_assign': variable name if the call returns a value, else None
    """
    cs = routine['callstatement']
    if cs is None:
        return None

    result = {
        'pre_call': [],
        'call_args': [],
        'post_call': [],
        'return_assign': None,
    }

    # Handle compound statements: {stmts; (*f2py_func)(...); stmts}
    if cs.startswith('{'):
        # Split into statements
        inner = cs[1:].rstrip('}').strip()
        stmts = _split_c_statements(inner)
        found_call = False
        for stmt in stmts:
            stmt = stmt.strip()
            if '(*f2py_func)' in stmt:
                found_call = True
                _parse_func_call(stmt, result)
            elif not found_call:
                result['pre_call'].append(stmt)
            else:
                result['post_call'].append(stmt)
    else:
        _parse_func_call(cs, result)

    return result


def _split_c_statements(s):
    """Split C code by semicolons, respecting braces and strings."""
    stmts = []
    depth = 0
    in_str = False
    current = []
    for ch in s:
        if ch == '"':
            in_str = not in_str
            current.append(ch)
        elif ch == '{' and not in_str:
            depth += 1
            current.append(ch)
        elif ch == '}' and not in_str:
            depth -= 1
            current.append(ch)
        elif ch == ';' and depth == 0 and not in_str:
            stmts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        stmts.append(''.join(current))
    return [s for s in stmts if s.strip()]


def _parse_func_call(stmt, result):
    """Parse a statement containing (*f2py_func)(args...) into result dict."""
    # Check for return value assignment
    assign_match = re.match(r'(\w+)\s*=\s*\(\*f2py_func\)', stmt)
    if assign_match:
        result['return_assign'] = assign_match.group(1)

    # Extract arguments from (*f2py_func)(...)
    # Find the opening paren after (*f2py_func)
    idx = stmt.find('(*f2py_func)')
    if idx == -1:
        return
    idx += len('(*f2py_func)')
    # Find the argument list
    if idx < len(stmt) and stmt[idx] == '(':
        args_str, _ = _extract_balanced_parens(stmt, idx)
        result['call_args'] = _split_respecting_parens_and_quotes(args_str)
    else:
        result['call_args'] = []


def _translate_callstatement_args(routine, cdef_param_types=None):
    """Translate callstatement arguments to Cython function call arguments."""
    parsed = _parse_callstatement(routine)
    if parsed is None:
        return None

    raw_args = parsed['call_args']

    # Check for wrapped complex return functions (fortranname ending in 'wrp').
    fortranname = routine.get('fortranname', '')
    if fortranname and 'wrp' in fortranname:
        if raw_args and raw_args[0].strip().startswith('&'):
            routine['_wrp_return_var'] = raw_args[0].strip()[1:]
            raw_args = raw_args[1:]

    # Map from cdef signature types for correct pointer casts
    _sig_to_cytype = {
        's': 'cy_s', 'd': 'cy_d', 'c': 'cy_c', 'z': 'cy_z',
        'int': 'blas_int', 'blas_int': 'blas_int', 'bint': 'blas_int',
    }

    translated = []
    for i, raw_arg in enumerate(raw_args):
        raw_arg = raw_arg.strip()
        # Get expected type from cdef signature
        sig_type = (cdef_param_types[i]
                    if cdef_param_types and i < len(cdef_param_types)
                    else None)
        cytype = _sig_to_cytype.get(sig_type) if sig_type else None
        translated.append(
            _translate_single_call_arg(raw_arg, routine, cytype)
        )

    return ', '.join(translated)


def _split_respecting_parens_and_quotes(s):
    """Split by comma, respecting parentheses and quoted strings."""
    parts = []
    depth = 0
    in_quote = False
    current = []
    for ch in s:
        if ch == '"' and not in_quote:
            in_quote = True
            current.append(ch)
        elif ch == '"' and in_quote:
            in_quote = False
            current.append(ch)
        elif ch == '(' and not in_quote:
            depth += 1
            current.append(ch)
        elif ch == ')' and not in_quote:
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0 and not in_quote:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


def _translate_single_call_arg(arg, routine, expected_cytype=None):
    """Translate a single callstatement argument to Cython.

    Parameters
    ----------
    arg : str
        Raw argument from the callstatement.
    routine : dict
        The routine specification.
    expected_cytype : str, optional
        Expected Cython typedef (cy_s, cy_d, cy_c, cy_z) from the cdef
        signature. Used for correct pointer casts.
    """
    arg = arg.strip()

    # String indexing: &"CHARS"[var]
    str_idx_match = re.match(r'&"([^"]+)"\[(\w+)\]', arg)
    if str_idx_match:
        chars = str_idx_match.group(1)
        var = str_idx_match.group(2)
        return f'(<char *>b"{chars}" + {var})'

    # Ternary char mapping: (cond?"X":"Y") or nested
    if '?' in arg and '"' in arg:
        return _translate_char_ternary(arg, routine)

    # &var - scalar by reference
    if arg.startswith('&'):
        varname = arg[1:]
        ainfo = routine['args'].get(varname, {})
        if _is_array_arg(ainfo):
            cytype = expected_cytype or _ctype_ptr(ainfo).rstrip(' *').strip()
            return f'<{cytype} *>np.PyArray_DATA({varname})'
        elif ainfo.get('ftype') == 'character':
            # Character args are Python bytes - cast to char*
            return f'<char *>{varname}'
        else:
            # Use expected type from cdef signature if available
            if expected_cytype:
                return f'<{expected_cytype} *>&{varname}'
            else:
                return f'&{varname}'

    # array+offset: var+off
    plus_match = re.match(r'(\w+)\+(\w+)', arg)
    if plus_match:
        varname = plus_match.group(1)
        offset = plus_match.group(2)
        ainfo = routine['args'].get(varname, {})
        if _is_array_arg(ainfo):
            cytype = expected_cytype or _ctype_ptr(ainfo).rstrip(' *').strip()
            return f'<{cytype} *>np.PyArray_DATA({varname}) + {offset}'
        else:
            return f'{varname} + {offset}'

    # Plain variable name (array passed directly, or char* variable)
    ainfo = routine['args'].get(arg, {})
    if _is_array_arg(ainfo):
        cytype = expected_cytype or _ctype_ptr(ainfo).rstrip(' *').strip()
        return f'<{cytype} *>np.PyArray_DATA({arg})'
    elif arg in routine['args']:
        ftype = ainfo.get('ftype')
        if ftype == 'character':
            # Character args are Python bytes - cast to char*
            return f'<char *>{arg}'
        elif expected_cytype:
            return f'<{expected_cytype} *>&{arg}'
        else:
            return f'&{arg}'
    else:
        return arg


def _translate_char_ternary(expr, routine):
    """Translate a C char* ternary to Cython.

    Handles various patterns of integer-to-char mappings used in f2py
    callstatements. Returns a Cython expression that produces a char*.

    Patterns handled:
    - (var?"X":"Y") -> &"YX"[var]
    - (var?(var==2?"C":"T"):"N") -> &"NTC"[var]
    - (var>0?(var==1?"T":"C"):"N") -> &"NTC"[var]
    - (a?(b?"X":"Y"):"Z") -> nested boolean
    - (a?(b?"X":"Y"):(c?"W":"Z")) -> double nested boolean
    """
    expr = expr.strip()

    # Pattern 1: Three-way via ==: (var?(var==2?"C":"T"):"N")
    m = re.match(
        r'\((\w+)\?\((\w+)==(\d+)\?"(.)":"(.)"\):"(.)"\)',
        expr
    )
    if m:
        var = m.group(1)
        val = int(m.group(3))
        char_eq = m.group(4)
        char_neq = m.group(5)
        char_false = m.group(6)
        if val == 2:
            lookup = char_false + char_neq + char_eq
            return f'(<char *>b"{lookup}" + {var})'

    # Pattern 2: Three-way via >0 and ==1: (var>0?(var==1?"T":"C"):"N")
    m = re.match(
        r'\((\w+)>0\?\((\w+)==(\d+)\?"(.)":"(.)"\):"(.)"\)',
        expr
    )
    if m:
        var = m.group(1)
        val = int(m.group(3))
        char_eq = m.group(4)
        char_neq = m.group(5)
        char_false = m.group(6)
        if val == 1:
            lookup = char_false + char_eq + char_neq
            return f'(<char *>b"{lookup}" + {var})'

    # Pattern 3: Nested boolean: (a?(b?"X":"Y"):"Z")
    m = re.match(
        r'\((\w+)\?\((\w+)\?"(.)":"(.)"\):"(.)"\)',
        expr
    )
    if m:
        var_a = m.group(1)
        var_b = m.group(2)
        char_bt = m.group(3)  # b true
        char_bf = m.group(4)  # b false
        char_af = m.group(5)  # a false
        return (f'((<char *>b"{char_bf}{char_bt}" + {var_b}) if {var_a} '
                f'else (<char *>b"{char_af}"))')

    # Pattern 4: Double nested: (a?(b?"X":"Y"):(c?"W":"Z"))
    m = re.match(
        r'\((\w+)\?\((\w+)\?"(.)":"(.)"\):\((\w+)\?"(.)":"(.)"\)\)',
        expr
    )
    if m:
        var_a = m.group(1)
        var_b = m.group(2)
        char_bt = m.group(3)
        char_bf = m.group(4)
        var_c = m.group(5)
        char_ct = m.group(6)
        char_cf = m.group(7)
        return (f'((<char *>b"{char_bf}{char_bt}" + {var_b}) if {var_a} '
                f'else (<char *>b"{char_cf}{char_ct}" + {var_c}))')

    # Pattern 5: Simple boolean: (var?"X":"Y")
    m = re.match(r'\((\w+)\?"(.)":"(.)"\)', expr)
    if m:
        var = m.group(1)
        char_true = m.group(2)
        char_false = m.group(3)
        return f'(<char *>b"{char_false}{char_true}" + {var})'

    # Fallback: leave as C comment for manual fix
    return f'/* FIXME char ternary: {expr} */'


def _build_return(routine):
    """Build return value info for a routine.

    Returns list of dicts with 'var' (the variable name in scope)
    and 'name' (the name f2py would use in the result, via out=X).

    f2py convention: pure intent(out) args first, then intent(in,out) args,
    in the order they appear in the argument list.
    """
    is_function = routine['type'] == 'function'

    return_parts = []
    seen = set()

    # For functions, the return value comes first
    if is_function and routine.get('result_name'):
        rn = routine['result_name']
        return_parts.append({'var': rn, 'name': rn})
        seen.add(rn)

    # Return outputs in arg_names order, with info always last.
    # This matches f2py's return order convention.
    for aname in routine['arg_names']:
        if aname == 'info':
            continue
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        out_name = ainfo.get('out_name', aname)
        if 'out' in intents:
            if out_name not in seen:
                return_parts.append({'var': aname, 'name': out_name})
                seen.add(out_name)

    # Also check for output args not in arg_names (e.g., extra output params)
    for aname, ainfo in routine['args'].items():
        if aname in seen or aname == 'info':
            continue
        intents = ainfo.get('intents', [])
        out_name = ainfo.get('out_name', aname)
        if 'out' in intents:
            if out_name not in seen:
                return_parts.append({'var': aname, 'name': out_name})
                seen.add(out_name)

    # info always last
    if 'info' in routine['args']:
        ainfo = routine['args']['info']
        if 'out' in ainfo.get('intents', []):
            return_parts.append({'var': 'info', 'name': 'info'})

    return return_parts


def _generate_post_call(routine):
    """Generate Python code for post-call processing from the callstatement.

    Many LAPACK wrappers have post-processing steps like:
    - Decrementing pivot indices from 1-based (Fortran) to 0-based (Python)
    - Decrementing scalar output values (hi--, lo--)
    - Copying workspace values to output arrays

    Returns a list of Python code lines (with leading '    ' indent).
    """
    cs = routine.get('callstatement')
    if not cs or not cs.startswith('{'):
        return []

    parsed = _parse_callstatement(routine)
    if not parsed or not parsed['post_call']:
        return []

    # Rejoin the post_call parts (they were split by ';' but for loops
    # use ';' internally too)
    raw = ';'.join(parsed['post_call']).strip().rstrip('}')

    lines = []

    # Pattern 1: for(i=0;i<N;--arr[i++]) - decrement array elements
    # This converts 1-based Fortran indices to 0-based Python
    m = re.search(r'for\(i=0;i<(\w+);--(\w+)\[i\+\+\]\)', raw)
    if m:
        limit = m.group(1)
        arr = m.group(2)
        lines.append(f'    if not {arr}.flags.writeable:')
        lines.append(f'        {arr} = {arr}.copy()')
        lines.append(f'    {arr} -= 1  # Convert from 1-based to 0-based indexing')

    # Pattern 1b: for(i=0,n=MIN(m,n);i<n;--arr[i++]) - with limit computation
    m = re.search(r'for\(i=0,\w+=MIN\((\w+),(\w+)\);i<\w+;--(\w+)\[i\+\+\]\)', raw)
    if m:
        arr = m.group(3)
        if not any(arr in l for l in lines):
            lines.append(f'    if not {arr}.flags.writeable:')
        lines.append(f'        {arr} = {arr}.copy()')
        lines.append(f'    {arr} -= 1  # Convert from 1-based to 0-based indexing')

    # Pattern 2: for(i=0;i<N;--arr1[i],--arr2[i++]) - decrement two arrays
    m = re.search(r'for\(i=0;i<\w+;--(\w+)\[i\],--(\w+)\[i\+\+\]\)', raw)
    if m:
        arr1 = m.group(1)
        arr2 = m.group(2)
        if not any(arr1 in l for l in lines):
            lines.append(f'    {arr1} -= 1  # Convert from 1-based to 0-based indexing')
        if not any(arr2 in l for l in lines):
            lines.append(f'    {arr2} -= 1  # Convert from 1-based to 0-based indexing')

    # Pattern 3: var-- - decrement scalar
    for m in re.finditer(r'\b(\w+)--', raw):
        var = m.group(1)
        # Skip loop variable 'i' and array indexing
        if var != 'i' and '[' not in raw[m.start():m.end()+5]:
            if not any(var in l for l in lines):
                lines.append(f'    {var} -= 1  # Convert from 1-based to 0-based')

    # Pattern 4: for(i=0;i<N;i++){out[i] = src[i];} - copy values
    for m in re.finditer(
        r'for\(i=0;i<(\d+);i\+\+\)\{(\w+)\[i\]\s*=\s*(\w+)\[i\];\}', raw
    ):
        count = m.group(1)
        dst = m.group(2)
        src = m.group(3)
        lines.append(f'    {dst}[:] = {src}[:{count}]')

    return lines


def _generate_lwork_wrapper(routine, lib_module_name, cdef_sigs):
    """Generate a _lwork helper function.

    These call the main LAPACK routine with lwork=-1 to query optimal
    workspace size. The routine writes the optimal size to work[0].
    """
    name = routine['name']
    base_name = name.replace('_lwork', '')

    # Get visible args
    py_args = _get_python_args(routine)

    # Determine primary float type
    primary_ftype = None
    for aname in routine['arg_names']:
        ainfo = routine['args'].get(aname, {})
        ft = ainfo.get('ftype')
        if ft and ft not in ('integer', 'logical', 'character'):
            primary_ftype = ft
            break
    if primary_ftype is None:
        primary_ftype = 'double precision'

    numpy_dtype = _get_numpy_dtype(primary_ftype)
    # Use the actual work arg's ftype for the work variable type
    work_ftype = routine['args'].get('work', {}).get('ftype', primary_ftype)
    ctype = FTYPE_TO_CTYPE.get(work_ftype, 'double')

    # Build signature - only the visible input args.
    # Skip work/lwork/info/iwork/liwork/rwork since they're handled internally.
    _lwork_internal = {'work', 'lwork', 'info', 'iwork', 'liwork', 'rwork'}
    sig_parts = []
    for aname in py_args:
        if aname in _lwork_internal:
            continue
        ainfo = routine['args'].get(aname, {})
        intents = ainfo.get('intents', [])
        # Skip output args
        if 'out' in intents and 'in' not in intents:
            continue
        default = ainfo.get('default')
        ftype = ainfo.get('ftype', 'integer')
        is_optional = ainfo.get('optional', False)
        if ftype in ('integer', 'logical'):
            if default is not None and _is_simple_literal(default):
                sig_parts.append(f'int {aname}={default}')
            elif default is not None:
                sig_parts.append(f'int {aname}=-1')
            else:
                sig_parts.append(f'int {aname}')
        elif ftype == 'character':
            if default:
                char_default = default.replace('"', '').replace("'", '')
                sig_parts.append(f'{aname}=b"{char_default}"')
            else:
                sig_parts.append(aname)
        elif _is_array_arg(ainfo):
            # Array args in _lwork: leave untyped (Python object)
            sig_parts.append(aname)
        else:
            ct = FTYPE_TO_CTYPE.get(ftype, 'double')
            if default is not None and _is_simple_literal(default):
                formatted = _format_literal(default, ftype)
                sig_parts.append(f'{ct} {aname}={formatted}')
            else:
                sig_parts.append(f'{ct} {aname}')

    sig = ', '.join(sig_parts)

    lines = []
    lines.append(f'def {name}({sig}):')
    lines.append(f'    """Workspace size query for ``{base_name}``."""')
    lines.append(f'    cdef:')
    lines.append(f'        blas_int lwork = -1')
    lines.append(f'        blas_int info = 0')
    lines.append(f'        {ctype} work')

    # Check for additional workspace queries
    has_liwork = any(a == 'liwork' for a in routine['arg_names'])
    if has_liwork:
        lines.append(f'        blas_int liwork = -1')
        lines.append(f'        blas_int iwork')

    # Check for rwork output (complex routines)
    rwork_is_output = ('rwork' in routine['args'] and
                       'out' in routine['args'].get('rwork', {}).get('intents', []))
    if rwork_is_output:
        # Determine real precision from routine name prefix
        # c-prefix (complex64) → float, z-prefix (complex128) → double
        # s-prefix (float32) → float, d-prefix (float64) → double
        real_ctype = 'float' if name[0] in ('c', 's') else 'double'
        lines.append(f'        {real_ctype} rwork')
        # Check for lrwork
        if 'lrwork' in routine['args']:
            lines.append(f'        blas_int lrwork = -1')

    iwork_is_output = ('iwork' in routine['args'] and
                       'out' in routine['args'].get('iwork', {}).get('intents', []))

    # Declare all hidden/dummy args needed by the callstatement.
    # In _lwork routines, ALL non-input args are dummies (even arrays).
    # Topologically sorted by dependencies.
    _already_declared = {'lwork', 'info', 'work'}
    if has_liwork:
        _already_declared.update({'liwork', 'iwork'})
    if rwork_is_output:
        _already_declared.add('rwork')
        if 'lrwork' in routine['args']:
            _already_declared.add('lrwork')
    hidden_sorted = _get_hidden_args(routine)
    for aname in hidden_sorted:
        if aname in _already_declared:
            continue
        ainfo = routine['args'].get(aname, {})
        ftype = ainfo.get('ftype')
        if ftype in ('integer', 'logical'):
            default = ainfo.get('default')
            if default:
                py_default = _translate_f2py_expr(default, routine['args'])
                lines.append(f'        blas_int {aname} = {py_default}')
            else:
                lines.append(f'        blas_int {aname} = 0')
        elif ftype == 'character':
            lines.append(f'        char {aname} = 0')
        elif ftype:
            ct = FTYPE_TO_CTYPE.get(ftype, 'double')
            lines.append(f'        {ct} {aname} = 0')
    # Also declare any non-hidden non-input args that appear in the
    # callstatement but aren't in the signature (e.g., iwork, rwork)
    for aname in routine['arg_names']:
        if aname in _already_declared:
            continue
        if aname in [s.split('=')[0].split()[-1] for s in sig_parts]:
            continue  # Already in signature
        if aname in [h for h in hidden_sorted]:
            continue  # Already declared above
        ainfo = routine['args'].get(aname, {})
        ftype = ainfo.get('ftype')
        if ftype in ('integer', 'logical'):
            lines.append(f'        blas_int {aname} = 0')
        elif ftype:
            ct = FTYPE_TO_CTYPE.get(ftype, 'double')
            lines.append(f'        {ct} {aname} = 0')

    # If the base routine exists in cdef_sigs, call it directly.
    # Otherwise fall back to NotImplementedError.
    if base_name not in cdef_sigs:
        lines.append(f'    raise NotImplementedError("{name}: {base_name} not in cython_lapack")')
        return '\n'.join(lines) + '\n'

    # Convert character args from str to bytes
    for aname in py_args:
        if aname in _lwork_internal:
            continue
        ainfo = routine['args'].get(aname, {})
        if ainfo.get('ftype') == 'character':
            lines.append(f'    if isinstance({aname}, str):')
            lines.append(f'        {aname} = {aname}.encode()')

    # Apply computed defaults (e.g., hi=n-1)
    for aname in py_args:
        if aname in _lwork_internal:
            continue
        ainfo = routine['args'].get(aname, {})
        default = ainfo.get('default')
        if default and not _is_simple_literal(default):
            py_expr = _translate_f2py_expr(default, routine['args'])
            if ainfo.get('ftype') in ('integer', 'logical'):
                lines.append(f'    if {aname} == -1:')
                lines.append(f'        {aname} = {py_expr}')

    # Parse the callstatement to build the call
    cs = routine.get('callstatement')
    if cs:
        # Handle pre-call statements (e.g., hi++; lo++)
        if cs.startswith('{'):
            parsed_cs = _parse_callstatement(routine)
            if parsed_cs and parsed_cs['pre_call']:
                for stmt in parsed_cs['pre_call']:
                    stmt = stmt.strip()
                    # var++ -> var += 1
                    m = re.match(r'(\w+)\+\+', stmt)
                    if m:
                        lines.append(f'    {m.group(1)} += 1')
                    # F_INT i=expr -> already handled in cdef block

        # Use the callstatement but call the base routine via cython_lapack
        sig_types = cdef_sigs.get(base_name, [])
        call_args = _translate_callstatement_args(routine, sig_types)
        if call_args:
            lines.append(f'    {lib_module_name}.{base_name}({call_args})')
            # Build return tuple following arg_names order for output args
            ret_parts = []
            for aname in routine['arg_names']:
                if aname == 'info':
                    continue
                ainfo = routine['args'].get(aname, {})
                if 'out' in ainfo.get('intents', []):
                    ret_parts.append(aname)
            ret_parts.append('info')
            lines.append(f'    return {", ".join(ret_parts)}')
            return '\n'.join(lines) + '\n'

    # Fallback
    lines.append(f'    raise NotImplementedError("{name} not yet implemented")')
    return '\n'.join(lines) + '\n'


def generate_blas_pyx(routines, ilp64=False):
    """Generate the _pyblas.pyx content."""
    cdef_sigs = _load_cdef_signatures('cython_blas_signatures.txt')
    available = set(cdef_sigs.keys())

    lines = [COMMENT_HEADER]
    lines.append('# cython: boundscheck = False')
    lines.append('# cython: wraparound = False')
    lines.append('# cython: cdivision = True')
    lines.append('')
    lines.append('import numpy as np')
    lines.append('cimport numpy as np')
    # Some BLAS routines (cspmv, cspr, csyr, zspmv, zspr, zsyr) are only
    # in LAPACK, not BLAS. Load LAPACK signatures too for fallback.
    lapack_sigs = _load_cdef_signatures('cython_lapack_signatures.txt')

    lines.append('from scipy.linalg cimport cython_blas, cython_lapack')
    lines.append('from scipy.linalg.cython_blas cimport (blas_int,')
    lines.append('    s as cy_s, d as cy_d, c as cy_c, z as cy_z)')
    lines.append('')
    lines.append('np.import_array()')
    lines.append('')
    lines.append('')
    lines.append('class error(Exception):')
    lines.append('    """BLAS error."""')
    lines.append('    pass')
    lines.append('')
    lines.append('# f2py compatibility alias')
    lines.append('__pyblas_error = error')
    lines.append('')
    lines.append('')

    skipped = []
    for routine in routines:
        name = routine['name']
        if name in available:
            code = _generate_wrapper_function(routine, 'cython_blas',
                                              cdef_sigs.get(name))
            lines.append(code)
            lines.append('')
        elif name in lapack_sigs:
            # Fallback: use cython_lapack for routines in LAPACK but not BLAS
            code = _generate_wrapper_function(routine, 'cython_lapack',
                                              lapack_sigs.get(name))
            lines.append(code)
            lines.append('')
        else:
            skipped.append(name)

    if skipped:
        lines.append(f'# Skipped {len(skipped)} routines not in cython_blas or cython_lapack:')
        lines.append(f'# {", ".join(sorted(skipped))}')
        lines.append('')

    # Add hand-written wrappers for routines not in .pyf.src files
    # but present in cython_blas (auto-wrapped by f2py from BLAS library)
    lines.append(_generate_extra_blas_wrappers())

    return '\n'.join(lines)


def _generate_gees_gges_wrappers():
    """Generate hand-written wrappers for gees/gges routines with callbacks."""
    return '''
# --- gees/gges routines with callback function support ---
# These routines take a user-defined eigenvalue selection function.
# We use module-level variables to pass the Python callable through
# to cdef callback functions.

import inspect as _inspect

def _get_nargs(func):
    """Get the number of positional parameters a callable accepts."""
    try:
        sig = _inspect.signature(func)
        return sum(1 for p in sig.parameters.values()
                   if p.default is _inspect.Parameter.empty
                   and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    except (ValueError, TypeError):
        return -1  # unknown

def _call_select2(func, a1, a2):
    """Call a 2-arg select function, falling back to 1-arg."""
    n = _get_nargs(func)
    if n == 1:
        return 1 if func(a1) else 0
    return 1 if func(a1, a2) else 0

def _call_select3(func, a1, a2, a3):
    """Call a 3-arg select function, falling back to 2 or 1."""
    n = _get_nargs(func)
    if n <= 1:
        return 1 if func(a1) else 0
    elif n == 2:
        return 1 if func(a1, a2) else 0
    return 1 if func(a1, a2, a3) else 0

cdef object _gees_select_callable = None

cdef blas_int _dselect2_callback(cy_d *arg1, cy_d *arg2) noexcept nogil:
    with gil:
        return _call_select2(<object>_gees_select_callable, arg1[0], arg2[0])

cdef blas_int _sselect2_callback(cy_s *arg1, cy_s *arg2) noexcept nogil:
    with gil:
        return _call_select2(<object>_gees_select_callable, arg1[0], arg2[0])

cdef blas_int _cselect1_callback(cy_c *arg) noexcept nogil:
    with gil:
        return 1 if (<object>_gees_select_callable)(arg[0]) else 0

cdef blas_int _zselect1_callback(cy_z *arg) noexcept nogil:
    with gil:
        return 1 if (<object>_gees_select_callable)(arg[0]) else 0

cdef object _gges_select_callable = None

cdef blas_int _dselect3_callback(cy_d *a1, cy_d *a2, cy_d *a3) noexcept nogil:
    with gil:
        return _call_select3(<object>_gges_select_callable, a1[0], a2[0], a3[0])

cdef blas_int _sselect3_callback(cy_s *a1, cy_s *a2, cy_s *a3) noexcept nogil:
    with gil:
        return _call_select3(<object>_gges_select_callable, a1[0], a2[0], a3[0])

cdef blas_int _cselect2_callback(cy_c *a1, cy_c *a2) noexcept nogil:
    with gil:
        return _call_select2(<object>_gges_select_callable, a1[0], a2[0])

cdef blas_int _zselect2_callback(cy_z *a1, cy_z *a2) noexcept nogil:
    with gil:
        return _call_select2(<object>_gges_select_callable, a1[0], a2[0])


def dgees(select, a, int compute_v=1, int sort_t=0, w=None, vs=None,
          int lwork=-1, int overwrite_a=0):
    """Wrapper for ``dgees``."""
    global _gees_select_callable
    cdef:
        blas_int n, nrows, ldvs, sdim, info

    _was_1d_a = (a is not None) and np.ndim(a) == 1
    if _was_1d_a:
        a = np.asarray(a).reshape(-1, 1)
    if not overwrite_a:
        a = np.array(a, dtype=np.float64, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.float64)

    n = a.shape[0]
    nrows = a.shape[0]
    ldvs = ((n if compute_v else 1))
    if lwork == -1:
        lwork = max(3*n, 1)

    wr = np.empty((n,), dtype=np.float64, order="F")
    wi = np.empty((n,), dtype=np.float64, order="F")
    vs = np.empty((ldvs, n), dtype=np.float64, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.float64, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    _gees_select_callable = select
    cython_lapack.dgees(
        (<char *>b"NV" + compute_v), (<char *>b"NS" + sort_t),
        &_dselect2_callback, &n,
        <cy_d *>np.PyArray_DATA(a), &nrows, &sdim,
        <cy_d *>np.PyArray_DATA(wr), <cy_d *>np.PyArray_DATA(wi),
        <cy_d *>np.PyArray_DATA(vs), &ldvs,
        <cy_d *>np.PyArray_DATA(work), &lwork,
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gees_select_callable = None

    if _was_1d_a:
        a = a.reshape(-1)
    return a, sdim, wr, wi, vs, work, info


def sgees(select, a, int compute_v=1, int sort_t=0, w=None, vs=None,
          int lwork=-1, int overwrite_a=0):
    """Wrapper for ``sgees``."""
    global _gees_select_callable
    cdef:
        blas_int n, nrows, ldvs, sdim, info

    _was_1d_a = (a is not None) and np.ndim(a) == 1
    if _was_1d_a:
        a = np.asarray(a).reshape(-1, 1)
    if not overwrite_a:
        a = np.array(a, dtype=np.float32, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.float32)

    n = a.shape[0]
    nrows = a.shape[0]
    ldvs = ((n if compute_v else 1))
    if lwork == -1:
        lwork = max(3*n, 1)

    wr = np.empty((n,), dtype=np.float32, order="F")
    wi = np.empty((n,), dtype=np.float32, order="F")
    vs = np.empty((ldvs, n), dtype=np.float32, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.float32, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    _gees_select_callable = select
    cython_lapack.sgees(
        (<char *>b"NV" + compute_v), (<char *>b"NS" + sort_t),
        &_sselect2_callback, &n,
        <cy_s *>np.PyArray_DATA(a), &nrows, &sdim,
        <cy_s *>np.PyArray_DATA(wr), <cy_s *>np.PyArray_DATA(wi),
        <cy_s *>np.PyArray_DATA(vs), &ldvs,
        <cy_s *>np.PyArray_DATA(work), &lwork,
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gees_select_callable = None

    if _was_1d_a:
        a = a.reshape(-1)
    return a, sdim, wr, wi, vs, work, info


def cgees(select, a, int compute_v=1, int sort_t=0, w=None, vs=None,
          int lwork=-1, int overwrite_a=0):
    """Wrapper for ``cgees``."""
    global _gees_select_callable
    cdef:
        blas_int n, nrows, ldvs, sdim, info

    _was_1d_a = (a is not None) and np.ndim(a) == 1
    if _was_1d_a:
        a = np.asarray(a).reshape(-1, 1)
    if not overwrite_a:
        a = np.array(a, dtype=np.complex64, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.complex64)

    n = a.shape[0]
    nrows = a.shape[0]
    ldvs = ((n if compute_v else 1))
    if lwork == -1:
        lwork = max(2*n, 1)

    w = np.empty((n,), dtype=np.complex64, order="F")
    vs = np.empty((ldvs, n), dtype=np.complex64, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.complex64, order="F")
    rwork = np.empty((n,), dtype=np.float32, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    _gees_select_callable = select
    cython_lapack.cgees(
        (<char *>b"NV" + compute_v), (<char *>b"NS" + sort_t),
        &_cselect1_callback, &n,
        <cy_c *>np.PyArray_DATA(a), &nrows, &sdim,
        <cy_c *>np.PyArray_DATA(w),
        <cy_c *>np.PyArray_DATA(vs), &ldvs,
        <cy_c *>np.PyArray_DATA(work), &lwork,
        <cy_s *>np.PyArray_DATA(rwork),
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gees_select_callable = None

    if _was_1d_a:
        a = a.reshape(-1)
    return a, sdim, w, vs, work, info


def zgees(select, a, int compute_v=1, int sort_t=0, w=None, vs=None,
          int lwork=-1, int overwrite_a=0):
    """Wrapper for ``zgees``."""
    global _gees_select_callable
    cdef:
        blas_int n, nrows, ldvs, sdim, info

    _was_1d_a = (a is not None) and np.ndim(a) == 1
    if _was_1d_a:
        a = np.asarray(a).reshape(-1, 1)
    if not overwrite_a:
        a = np.array(a, dtype=np.complex128, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.complex128)

    n = a.shape[0]
    nrows = a.shape[0]
    ldvs = ((n if compute_v else 1))
    if lwork == -1:
        lwork = max(2*n, 1)

    w = np.empty((n,), dtype=np.complex128, order="F")
    vs = np.empty((ldvs, n), dtype=np.complex128, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.complex128, order="F")
    rwork = np.empty((n,), dtype=np.float64, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    _gees_select_callable = select
    cython_lapack.zgees(
        (<char *>b"NV" + compute_v), (<char *>b"NS" + sort_t),
        &_zselect1_callback, &n,
        <cy_z *>np.PyArray_DATA(a), &nrows, &sdim,
        <cy_z *>np.PyArray_DATA(w),
        <cy_z *>np.PyArray_DATA(vs), &ldvs,
        <cy_z *>np.PyArray_DATA(work), &lwork,
        <cy_d *>np.PyArray_DATA(rwork),
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gees_select_callable = None

    if _was_1d_a:
        a = a.reshape(-1)
    return a, sdim, w, vs, work, info


def dgges(select, a, b, int compute_vl=1, int compute_vr=1,
          int lwork=-1, int overwrite_a=0, int overwrite_b=0, int sort_t=-1):
    """Wrapper for ``dgges``."""
    global _gges_select_callable
    cdef:
        blas_int n, lda, ldb, sdim, ldvsl, ldvsr, info

    if not overwrite_a:
        a = np.array(a, dtype=np.float64, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.float64)
    if not overwrite_b:
        b = np.array(b, dtype=np.float64, order="F", copy=True)
    else:
        b = np.asfortranarray(b, dtype=np.float64)

    n = a.shape[0]
    lda = max(1, n)
    ldb = max(1, n)
    ldvsl = ((n if compute_vl else 1))
    ldvsr = ((n if compute_vr else 1))
    if lwork == -1:
        lwork = max(8*n+16, 1)

    alphar = np.empty((n,), dtype=np.float64, order="F")
    alphai = np.empty((n,), dtype=np.float64, order="F")
    beta = np.empty((n,), dtype=np.float64, order="F")
    vsl = np.empty((ldvsl, n), dtype=np.float64, order="F")
    vsr = np.empty((ldvsr, n), dtype=np.float64, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.float64, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    if sort_t == -1:
        sort_t = 1 if select is not None else 0
    _gges_select_callable = select if select is not None else (lambda a, b, c: True)
    cython_lapack.dgges(
        (<char *>b"NV" + compute_vl), (<char *>b"NV" + compute_vr),
        (<char *>b"NS" + sort_t),
        &_dselect3_callback, &n,
        <cy_d *>np.PyArray_DATA(a), &lda,
        <cy_d *>np.PyArray_DATA(b), &ldb, &sdim,
        <cy_d *>np.PyArray_DATA(alphar), <cy_d *>np.PyArray_DATA(alphai),
        <cy_d *>np.PyArray_DATA(beta),
        <cy_d *>np.PyArray_DATA(vsl), &ldvsl,
        <cy_d *>np.PyArray_DATA(vsr), &ldvsr,
        <cy_d *>np.PyArray_DATA(work), &lwork,
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gges_select_callable = None

    return a, b, sdim, alphar, alphai, beta, vsl, vsr, work, info


def sgges(select, a, b, int compute_vl=1, int compute_vr=1,
          int lwork=-1, int overwrite_a=0, int overwrite_b=0, int sort_t=-1):
    """Wrapper for ``sgges``."""
    global _gges_select_callable
    cdef:
        blas_int n, lda, ldb, sdim, ldvsl, ldvsr, info

    if not overwrite_a:
        a = np.array(a, dtype=np.float32, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.float32)
    if not overwrite_b:
        b = np.array(b, dtype=np.float32, order="F", copy=True)
    else:
        b = np.asfortranarray(b, dtype=np.float32)

    n = a.shape[0]
    lda = max(1, n)
    ldb = max(1, n)
    ldvsl = ((n if compute_vl else 1))
    ldvsr = ((n if compute_vr else 1))
    if lwork == -1:
        lwork = max(8*n+16, 1)

    alphar = np.empty((n,), dtype=np.float32, order="F")
    alphai = np.empty((n,), dtype=np.float32, order="F")
    beta = np.empty((n,), dtype=np.float32, order="F")
    vsl = np.empty((ldvsl, n), dtype=np.float32, order="F")
    vsr = np.empty((ldvsr, n), dtype=np.float32, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.float32, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    if sort_t == -1:
        sort_t = 1 if select is not None else 0
    _gges_select_callable = select if select is not None else (lambda a, b, c: True)
    cython_lapack.sgges(
        (<char *>b"NV" + compute_vl), (<char *>b"NV" + compute_vr),
        (<char *>b"NS" + sort_t),
        &_sselect3_callback, &n,
        <cy_s *>np.PyArray_DATA(a), &lda,
        <cy_s *>np.PyArray_DATA(b), &ldb, &sdim,
        <cy_s *>np.PyArray_DATA(alphar), <cy_s *>np.PyArray_DATA(alphai),
        <cy_s *>np.PyArray_DATA(beta),
        <cy_s *>np.PyArray_DATA(vsl), &ldvsl,
        <cy_s *>np.PyArray_DATA(vsr), &ldvsr,
        <cy_s *>np.PyArray_DATA(work), &lwork,
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gges_select_callable = None

    return a, b, sdim, alphar, alphai, beta, vsl, vsr, work, info


def cgges(select, a, b, int compute_vl=1, int compute_vr=1,
          int lwork=-1, int overwrite_a=0, int overwrite_b=0, int sort_t=-1):
    """Wrapper for ``cgges``."""
    global _gges_select_callable
    cdef:
        blas_int n, lda, ldb, sdim, ldvsl, ldvsr, info

    if not overwrite_a:
        a = np.array(a, dtype=np.complex64, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.complex64)
    if not overwrite_b:
        b = np.array(b, dtype=np.complex64, order="F", copy=True)
    else:
        b = np.asfortranarray(b, dtype=np.complex64)

    n = a.shape[0]
    lda = max(1, n)
    ldb = max(1, n)
    ldvsl = ((n if compute_vl else 1))
    ldvsr = ((n if compute_vr else 1))
    if lwork == -1:
        lwork = max(2*n, 1)

    alpha = np.empty((n,), dtype=np.complex64, order="F")
    beta = np.empty((n,), dtype=np.complex64, order="F")
    vsl = np.empty((ldvsl, n), dtype=np.complex64, order="F")
    vsr = np.empty((ldvsr, n), dtype=np.complex64, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.complex64, order="F")
    rwork = np.empty((8*n,), dtype=np.float32, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    if sort_t == -1:
        sort_t = 1 if select is not None else 0
    _gges_select_callable = select if select is not None else (lambda a, b: True)
    cython_lapack.cgges(
        (<char *>b"NV" + compute_vl), (<char *>b"NV" + compute_vr),
        (<char *>b"NS" + sort_t),
        &_cselect2_callback, &n,
        <cy_c *>np.PyArray_DATA(a), &lda,
        <cy_c *>np.PyArray_DATA(b), &ldb, &sdim,
        <cy_c *>np.PyArray_DATA(alpha),
        <cy_c *>np.PyArray_DATA(beta),
        <cy_c *>np.PyArray_DATA(vsl), &ldvsl,
        <cy_c *>np.PyArray_DATA(vsr), &ldvsr,
        <cy_c *>np.PyArray_DATA(work), &lwork,
        <cy_s *>np.PyArray_DATA(rwork),
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gges_select_callable = None

    return a, b, sdim, alpha, beta, vsl, vsr, work, info


def zgges(select, a, b, int compute_vl=1, int compute_vr=1,
          int lwork=-1, int overwrite_a=0, int overwrite_b=0, int sort_t=-1):
    """Wrapper for ``zgges``."""
    global _gges_select_callable
    cdef:
        blas_int n, lda, ldb, sdim, ldvsl, ldvsr, info

    if not overwrite_a:
        a = np.array(a, dtype=np.complex128, order="F", copy=True)
    else:
        a = np.asfortranarray(a, dtype=np.complex128)
    if not overwrite_b:
        b = np.array(b, dtype=np.complex128, order="F", copy=True)
    else:
        b = np.asfortranarray(b, dtype=np.complex128)

    n = a.shape[0]
    lda = max(1, n)
    ldb = max(1, n)
    ldvsl = ((n if compute_vl else 1))
    ldvsr = ((n if compute_vr else 1))
    if lwork == -1:
        lwork = max(2*n, 1)

    alpha = np.empty((n,), dtype=np.complex128, order="F")
    beta = np.empty((n,), dtype=np.complex128, order="F")
    vsl = np.empty((ldvsl, n), dtype=np.complex128, order="F")
    vsr = np.empty((ldvsr, n), dtype=np.complex128, order="F")
    work = np.empty((max(lwork, 1),), dtype=np.complex128, order="F")
    rwork = np.empty((8*n,), dtype=np.float64, order="F")
    bwork = np.empty((n,), dtype=np.intc, order="F")

    if sort_t == -1:
        sort_t = 1 if select is not None else 0
    _gges_select_callable = select if select is not None else (lambda a, b: True)
    cython_lapack.zgges(
        (<char *>b"NV" + compute_vl), (<char *>b"NV" + compute_vr),
        (<char *>b"NS" + sort_t),
        &_zselect2_callback, &n,
        <cy_z *>np.PyArray_DATA(a), &lda,
        <cy_z *>np.PyArray_DATA(b), &ldb, &sdim,
        <cy_z *>np.PyArray_DATA(alpha),
        <cy_z *>np.PyArray_DATA(beta),
        <cy_z *>np.PyArray_DATA(vsl), &ldvsl,
        <cy_z *>np.PyArray_DATA(vsr), &ldvsr,
        <cy_z *>np.PyArray_DATA(work), &lwork,
        <cy_d *>np.PyArray_DATA(rwork),
        <blas_int *>np.PyArray_DATA(bwork), &info)
    _gges_select_callable = None

    return a, b, sdim, alpha, beta, vsl, vsr, work, info
'''


def _generate_extra_blas_wrappers():
    """Generate wrappers for BLAS routines not in the .pyf.src files.

    These routines (dspr2, chpr2, zhpr2) are in the BLAS library and
    auto-wrapped by f2py, but not explicitly defined in the pyf files.
    """
    return '''
# --- Extra BLAS routines not in .pyf.src files ---
# These are in cython_blas but not in the f2py interface files.
# f2py auto-wraps them from the linked BLAS library.

def dspr2(int n, double alpha, x, y, ap, int incx=1, int offx=0,
          int incy=1, int offy=0, int lower=0, int overwrite_ap=0):
    """Wrapper for ``dspr2``."""
    x = np.asfortranarray(x, dtype=np.float64)
    y = np.asfortranarray(y, dtype=np.float64)
    if not overwrite_ap:
        ap = np.array(ap, dtype=np.float64, order="F", copy=True)
    else:
        ap = np.asfortranarray(ap, dtype=np.float64)
    cython_blas.dspr2((<char *>b"UL" + lower), &n,
                      <cy_d *>&alpha, <cy_d *>np.PyArray_DATA(x) + offx, &incx,
                      <cy_d *>np.PyArray_DATA(y) + offy, &incy,
                      <cy_d *>np.PyArray_DATA(ap))
    return ap


def chpr2(int n, float complex alpha, x, y, ap, int incx=1, int offx=0,
          int incy=1, int offy=0, int lower=0, int overwrite_ap=0):
    """Wrapper for ``chpr2``."""
    x = np.asfortranarray(x, dtype=np.complex64)
    y = np.asfortranarray(y, dtype=np.complex64)
    if not overwrite_ap:
        ap = np.array(ap, dtype=np.complex64, order="F", copy=True)
    else:
        ap = np.asfortranarray(ap, dtype=np.complex64)
    cython_blas.chpr2((<char *>b"UL" + lower), &n,
                      <cy_c *>&alpha, <cy_c *>np.PyArray_DATA(x) + offx, &incx,
                      <cy_c *>np.PyArray_DATA(y) + offy, &incy,
                      <cy_c *>np.PyArray_DATA(ap))
    return ap


def zhpr2(int n, double complex alpha, x, y, ap, int incx=1, int offx=0,
          int incy=1, int offy=0, int lower=0, int overwrite_ap=0):
    """Wrapper for ``zhpr2``."""
    x = np.asfortranarray(x, dtype=np.complex128)
    y = np.asfortranarray(y, dtype=np.complex128)
    if not overwrite_ap:
        ap = np.array(ap, dtype=np.complex128, order="F", copy=True)
    else:
        ap = np.asfortranarray(ap, dtype=np.complex128)
    cython_blas.zhpr2((<char *>b"UL" + lower), &n,
                      <cy_z *>&alpha, <cy_z *>np.PyArray_DATA(x) + offx, &incx,
                      <cy_z *>np.PyArray_DATA(y) + offy, &incy,
                      <cy_z *>np.PyArray_DATA(ap))
    return ap
'''


def generate_lapack_pyx(routines, ilp64=False):
    """Generate the _pylapack.pyx content."""
    cdef_sigs = _load_cdef_signatures('cython_lapack_signatures.txt')
    available = set(cdef_sigs.keys())

    lines = [COMMENT_HEADER]
    lines.append('# cython: boundscheck = False')
    lines.append('# cython: wraparound = False')
    lines.append('# cython: cdivision = True')
    lines.append('')
    lines.append('import numpy as np')
    lines.append('cimport numpy as np')
    lines.append('from scipy.linalg cimport cython_lapack')
    lines.append('from scipy.linalg.cython_lapack cimport (blas_int,')
    lines.append('    s as cy_s, d as cy_d, c as cy_c, z as cy_z,')
    lines.append('    sselect2, sselect3, dselect2, dselect3,')
    lines.append('    cselect1, cselect2, zselect1, zselect2)')
    lines.append('')
    lines.append('np.import_array()')
    lines.append('')
    lines.append('')

    # gees/gges callback routines are hand-written, not auto-generated
    _callback_routines = {
        'cgees', 'dgees', 'sgees', 'zgees',
        'cgges', 'dgges', 'sgges', 'zgges',
    }

    skipped = []
    lwork_count = 0
    for routine in routines:
        name = routine['name']

        # Skip routines with callback function arguments
        if name in _callback_routines:
            skipped.append(name)
            continue

        # _lwork helpers: these are artificial f2py routines that query
        # workspace size by calling the main routine with lwork=-1
        if '_lwork' in name:
            code = _generate_lwork_wrapper(routine, 'cython_lapack', cdef_sigs)
            lines.append(code)
            lines.append('')
            lwork_count += 1
            continue

        if name not in available:
            skipped.append(name)
            continue

        code = _generate_wrapper_function(routine, 'cython_lapack',
                                          cdef_sigs.get(name))
        lines.append(code)
        lines.append('')

    if skipped:
        lines.append(f'# Skipped {len(skipped)} routines not in cython_lapack:')
        lines.append(f'# {", ".join(sorted(skipped))}')
        lines.append('')

    # Add hand-written gees/gges wrappers with callback support
    lines.append(_generate_gees_gges_wrappers())

    return '\n'.join(lines)


def main():
    parser_module = _import_module_from_file(
        '_pyf_parser',
        os.path.join(BASE_DIR, '_pyf_parser.py')
    )

    ap = argparse.ArgumentParser(description='Generate Python BLAS/LAPACK wrappers')
    ap.add_argument('-o', '--outdir', required=True,
                    help='Output directory')
    ap.add_argument('--ilp64', action='store_true',
                    help='Generate ILP64 wrappers')
    ap.add_argument('--blas-only', action='store_true',
                    help='Only generate BLAS wrappers')
    ap.add_argument('--lapack-only', action='store_true',
                    help='Only generate LAPACK wrappers')
    ap.add_argument('--dump', action='store_true',
                    help='Dump generated code to stdout instead of files')
    args = ap.parse_args()

    suffix = '_64' if args.ilp64 else ''

    if not args.lapack_only:
        blas_src = os.path.join(BASE_DIR, 'fblas.pyf.src')
        print(f'Parsing {blas_src}...')
        blas_routines = parser_module.parse_pyf_file(blas_src)
        print(f'  -> {len(blas_routines)} BLAS routines')

        blas_pyx = generate_blas_pyx(blas_routines, ilp64=args.ilp64)

        if args.dump:
            print(blas_pyx)
        else:
            outfile = os.path.join(args.outdir, f'_pyblas{suffix}.pyx')
            os.makedirs(args.outdir, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write(blas_pyx)
            print(f'Wrote {outfile}')

    if not args.blas_only:
        lapack_src = os.path.join(BASE_DIR, 'flapack.pyf.src')
        print(f'Parsing {lapack_src}...')
        lapack_routines = parser_module.parse_pyf_file(lapack_src)
        print(f'  -> {len(lapack_routines)} LAPACK routines')

        lapack_pyx = generate_lapack_pyx(lapack_routines, ilp64=args.ilp64)

        if args.dump:
            print(lapack_pyx)
        else:
            outfile = os.path.join(args.outdir, f'_pylapack{suffix}.pyx')
            os.makedirs(args.outdir, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write(lapack_pyx)
            print(f'Wrote {outfile}')


if __name__ == '__main__':
    main()
