"""
Parser for f2py .pyf interface files (after template expansion).

Parses the expanded .pyf files into structured Python dicts describing
each routine's signature, arguments, intents, checks, callstatements, etc.
This is used by the Cython wrapper generator to produce Python-facing
wrappers that replace f2py-generated modules.

Usage::

    from scipy.linalg._pyf_parser import parse_pyf_file
    routines = parse_pyf_file('scipy/linalg/fblas.pyf.src')
"""

import os
import re
import sys


# ---------------------------------------------------------------------------
# Template expansion (vendored from tools/generate_f2pymod.py)
# ---------------------------------------------------------------------------

routine_start_re = re.compile(
    r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b', re.I
)
routine_end_re = re.compile(
    r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I
)
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)


def _parse_structure(astr):
    spanlist = []
    ind = 0
    while True:
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
            while True:
                i = astr.rfind('\n', ind, start)
                if i == -1:
                    break
                start = i
                if astr[i:i+7] != '\n     $':
                    break
        start += 1
        m = routine_end_re.search(astr, m.end())
        ind = end = m and m.end() - 1 or len(astr)
        spanlist.append((start, end))
    return spanlist


template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
list_re = re.compile(r"<\s*((.*?))\s*>")
item_re = re.compile(r"\A\\(?P<index>\d+)\Z")


def _conv(astr):
    b = astr.split(',')
    lst = [x.strip() for x in b]
    for i in range(len(lst)):
        m = item_re.match(lst[i])
        if m:
            j = int(m.group('index'))
            lst[i] = lst[j]
    return ','.join(lst)


def _find_repl_patterns(astr):
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        name = rep[0].strip() or _unique_key(names)
        repl = rep[1].replace(r'\,', '@comma@')
        thelist = _conv(repl)
        names[name] = thelist
    return names


def _find_and_remove_repl_patterns(astr):
    names = _find_repl_patterns(astr)
    astr = re.subn(named_re, '', astr)[0]
    return astr, names


def _unique_key(adict):
    allkeys = list(adict.keys())
    n = 1
    while True:
        newkey = f'__l{n}'
        if newkey not in allkeys:
            return newkey
        n += 1


template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')


def _expand_sub(substr, names):
    substr = substr.replace(r'\>', '@rightarrow@')
    substr = substr.replace(r'\<', '@leftarrow@')
    lnames = _find_repl_patterns(substr)
    substr = named_re.sub(r"<\1>", substr)

    def listrepl(mobj):
        thelist = _conv(mobj.group(1).replace(r'\,', '@comma@'))
        if template_name_re.match(thelist):
            return f"<{thelist}>"
        name = None
        for key in lnames.keys():
            if lnames[key] == thelist:
                name = key
        if name is None:
            name = _unique_key(lnames)
            lnames[name] = thelist
        return f"<{name}>"

    substr = list_re.sub(listrepl, substr)

    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if r not in rules:
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError(f'No replicates found for <{r}>')
            if r not in names and not thelist.startswith('_'):
                names[r] = thelist
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)
            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                pass  # mismatch, ignore
    if not rules:
        return substr

    def namerepl(mobj):
        name = mobj.group(1)
        return rules.get(name, (k+1)*[name])[k]

    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    return newstr


_special_names = _find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

include_src_re = re.compile(
    r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+\.src)['\"]", re.I
)


def _resolve_includes(source):
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    lines.extend(_resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def expand_pyf_src(source):
    """Expand a .pyf.src file: resolve includes and expand templates.

    Returns the fully expanded .pyf content as a string.
    """
    lines = _resolve_includes(source)
    allstr = ''.join(lines)

    newstr = allstr
    writestr = ''
    struct = _parse_structure(newstr)
    oldend = 0
    names = {}
    names.update(_special_names)
    for sub in struct:
        cleanedstr, defs = _find_and_remove_repl_patterns(
            newstr[oldend:sub[0]]
        )
        writestr += cleanedstr
        names.update(defs)
        writestr += _expand_sub(newstr[sub[0]:sub[1]], names)
        oldend = sub[1]
    writestr += newstr[oldend:]
    return writestr


# ---------------------------------------------------------------------------
# .pyf block parser
# ---------------------------------------------------------------------------

# Fortran type to numpy dtype mapping
_ftype_to_dtype = {
    'real': 'float32',
    'double precision': 'float64',
    'complex': 'complex64',
    'double complex': 'complex128',
    'integer': 'int',  # will be blas_int
    'logical': 'int',
}

# Fortran type to C type mapping
_ftype_to_ctype = {
    'real': 'float',
    'double precision': 'double',
    'complex': 'complex_float',
    'double complex': 'complex_double',
    'integer': 'F_INT',
}


def _parse_routine_block(block_text):
    """Parse a single subroutine/function block from expanded .pyf content.

    Parameters
    ----------
    block_text : str
        The text of a single subroutine or function block, from the
        "subroutine/function" keyword to "end subroutine/function".

    Returns
    -------
    routine : dict
        Structured description of the routine.
    """
    lines = block_text.strip().split('\n')
    # Join continuation lines (lines ending with &, or starting with &)
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip empty lines and pure comment lines
        if not line or line.startswith('!'):
            i += 1
            continue
        # Handle continuation: if line ends with &, join with next
        while line.endswith('&'):
            i += 1
            if i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith('!'):
                    i += 1
                    continue
                line = line[:-1].strip() + ' ' + next_line.lstrip('& ')
            else:
                line = line[:-1].strip()
                break
        joined_lines.append(line)
        i += 1

    if not joined_lines:
        return None

    # Parse the first line: subroutine/function declaration
    first_line = joined_lines[0]

    # Check for function with return type prefix
    func_match = re.match(
        r'(?:(\w[\w\s]*?)\s+)?'  # optional return type
        r'(subroutine|function)\s+'  # keyword
        r'(\w+)'  # name
        r'\s*\(([^)]*)\)'  # argument list
        r'(?:\s+result\s*\((\w+)\))?',  # optional result clause
        first_line, re.I
    )
    if not func_match:
        return None

    return_type_prefix = func_match.group(1)
    routine_type = func_match.group(2).lower()
    name = func_match.group(3)
    args_str = func_match.group(4)
    result_name = func_match.group(5)

    # Parse argument names from the declaration
    arg_names = [a.strip() for a in args_str.split(',') if a.strip()]

    routine = {
        'name': name,
        'type': routine_type,
        'return_type': return_type_prefix.strip() if return_type_prefix else None,
        'result_name': result_name,
        'arg_names': arg_names,
        'callstatement': None,
        'callprotoargument': None,
        'fortranname': None,
        'intent_c': None,  # intent(c) declaration for function return
        'docstring': '',
        'args': {},
    }

    # Process remaining lines
    i = 1
    while i < len(joined_lines):
        line = joined_lines[i]

        # Skip end statement
        if re.match(r'end\s+(subroutine|function)', line, re.I):
            break

        # Skip lines inside usercode blocks
        if line.lower().startswith('usercode'):
            i += 1
            while i < len(joined_lines):
                if joined_lines[i].strip().lower().startswith('end usercode'):
                    break
                i += 1
            i += 1
            continue

        # Docstring (comment lines starting with !)
        if line.startswith('!'):
            routine['docstring'] += line[1:].strip() + '\n'
            i += 1
            continue

        # callstatement
        cs_match = re.match(r'callstatement\s+(.*)', line, re.I)
        if cs_match:
            routine['callstatement'] = cs_match.group(1).strip()
            i += 1
            continue

        # callprotoargument
        cpa_match = re.match(r'callprotoargument\s+(.*)', line, re.I)
        if cpa_match:
            routine['callprotoargument'] = cpa_match.group(1).strip()
            i += 1
            continue

        # fortranname
        fn_match = re.match(r'fortranname\s+(.*)', line, re.I)
        if fn_match:
            routine['fortranname'] = fn_match.group(1).strip()
            i += 1
            continue

        # intent(c) for function name
        ic_match = re.match(r'intent\(c\)\s+(\w+)', line, re.I)
        if ic_match:
            routine['intent_c'] = ic_match.group(1)
            i += 1
            continue

        # Skip standalone keywords
        if line.lower().strip() in ('threadsafe',):
            i += 1
            continue

        # Variable declaration or check/depend line
        # Could be: "<type> [attrs] :: <varlist> [= default]"
        # Or: "check(...) :: <varname>"
        _parse_var_line(line, routine)
        i += 1

    return routine


def _parse_intent(intent_str):
    """Parse an intent string like 'in,out,copy,out=lu' into a dict.

    Returns
    -------
    result : dict with keys:
        'intents': list of str (e.g., ['in', 'out', 'copy'])
        'out_name': str or None (e.g., 'lu' from 'out=lu')
    """
    intents = []
    out_name = None
    for part in intent_str.split(','):
        part = part.strip()
        if part.startswith('out='):
            out_name = part[4:].strip()
            if 'out' not in intents:
                intents.append('out')
        else:
            intents.append(part)
    return {'intents': intents, 'out_name': out_name}


def _parse_var_line(line, routine):
    """Parse a variable declaration or attribute line and update the routine dict."""

    # Standalone check: "check(expr) :: varname"
    if line.lower().startswith('check('):
        # Use balanced paren extraction
        content, end = _extract_balanced_parens(line, line.index('('))
        rest = line[end:].strip()
        # rest should be ":: varname"
        m = re.match(r'::\s*(\w+)', rest)
        if m:
            varname = m.group(1)
            if varname in routine['args']:
                routine['args'][varname].setdefault('checks', []).append(content)
            else:
                routine['args'].setdefault(varname, {}).setdefault(
                    'checks', []
                ).append(content)
            return

    # Standalone intent/attribute line WITHOUT '::': "intent(in,out,copy,out=lu) a"
    # This pattern occurs when attributes are added to already-declared variables
    if not '::' in line:
        attr_no_sep = re.match(
            r'((?:intent|depend|check|dimension|optional|threadsafe)\b.+?)\s+(\w+)\s*$',
            line, re.I
        )
        if attr_no_sep:
            attrs_str = attr_no_sep.group(1)
            varname = attr_no_sep.group(2)
            if varname not in routine['args']:
                routine['args'][varname] = {}
            _apply_attributes(attrs_str, routine['args'][varname])
            return

    # Type declaration: "<type> [attributes] :: <varlist>"
    # or without :: separator: "<type> <varlist>"  (common for function return types)
    # Fortran types can be multi-word: "double precision", "double complex"
    type_pattern = (
        r'(real|double\s+precision|complex|double\s+complex|'
        r'complex\*16|integer|logical|character)'
    )

    decl_match = re.match(
        type_pattern + r'(.*?)\s*::\s*(.*)', line, re.I
    )
    if not decl_match:
        # Try without :: separator: "<type> <varlist>"
        decl_no_sep = re.match(
            type_pattern + r'\s+([\w,\s]+)$', line, re.I
        )
        if decl_no_sep:
            ftype = re.sub(r'\s+', ' ', decl_no_sep.group(1).strip().lower())
            vars_str = decl_no_sep.group(2).strip()
            var_decls = _split_var_declarations(vars_str)
            for var_decl in var_decls:
                varname = var_decl['name']
                default = var_decl.get('default')
                if varname in routine['args']:
                    arg = routine['args'][varname]
                else:
                    arg = {}
                    routine['args'][varname] = arg
                arg['ftype'] = ftype
                arg['dtype'] = _ftype_to_dtype.get(ftype, ftype)
                if default is not None:
                    arg['default'] = default
            return
        # Try matching just attributes for existing vars: "intent(...) :: var"
        # or "depend(...) :: var"
        attr_match = re.match(r'((?:intent|depend|check|dimension)\(.+?\).*?)\s*::\s*(.*)', line, re.I)
        if attr_match:
            attrs_str = attr_match.group(1)
            vars_str = attr_match.group(2)
            # Parse the variable names
            for var_decl in _split_var_declarations(vars_str):
                varname = var_decl['name']
                if varname not in routine['args']:
                    routine['args'][varname] = {}
                _apply_attributes(attrs_str, routine['args'][varname])
            return
        return

    ftype = re.sub(r'\s+', ' ', decl_match.group(1).strip().lower())
    attrs_str = decl_match.group(2).strip()
    vars_str = decl_match.group(3).strip()

    # Strip the comma separating attrs from var declarations
    attrs_str = attrs_str.strip().rstrip(',').strip()

    # Parse variable declarations (may have "= default")
    var_decls = _split_var_declarations(vars_str)

    for var_decl in var_decls:
        varname = var_decl['name']
        default = var_decl.get('default')

        if varname in routine['args']:
            # Merge with existing placeholder
            arg = routine['args'][varname]
        else:
            arg = {}
            routine['args'][varname] = arg

        arg['ftype'] = ftype
        arg['dtype'] = _ftype_to_dtype.get(ftype, ftype)
        if default is not None:
            arg['default'] = default

        _apply_attributes(attrs_str, arg)


def _split_var_declarations(vars_str):
    """Split a comma-separated variable declaration list.

    Handles cases like: "a, b", "a = 0", "func_name, result_var"
    """
    result = []
    # Simple split by comma, but respect parentheses
    parts = _split_respecting_parens(vars_str, ',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check for "varname = default"
        eq_match = re.match(r'(\w+)\s*=\s*(.*)', part)
        if eq_match:
            result.append({
                'name': eq_match.group(1),
                'default': eq_match.group(2).strip()
            })
        else:
            # Just a variable name (possibly with dimension after it, but
            # dimension is usually in attrs)
            name_match = re.match(r'(\w+)', part)
            if name_match:
                result.append({'name': name_match.group(1)})
    return result


def _split_respecting_parens(s, sep):
    """Split string by separator, but don't split inside parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


def _extract_balanced_parens(s, start):
    """Extract a balanced parenthesized substring from position `start`.

    `s[start]` must be '('. Returns the content inside (excluding outer parens)
    and the index after the closing ')'.
    """
    assert s[start] == '('
    depth = 0
    i = start
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return s[start+1:i], i + 1
        i += 1
    # Unbalanced - return rest of string
    return s[start+1:], len(s)


def _find_keyword_args(attrs_str, keyword):
    """Find all instances of `keyword(...)` in attrs_str, handling nested parens.

    Returns list of content strings inside the parens.
    """
    results = []
    kw_lower = keyword.lower()
    i = 0
    while i < len(attrs_str):
        # Find keyword
        idx = attrs_str.lower().find(kw_lower, i)
        if idx == -1:
            break
        # Check it's the whole keyword (not part of a larger word)
        if idx > 0 and attrs_str[idx-1].isalnum():
            i = idx + 1
            continue
        # Find the opening paren
        paren_idx = idx + len(keyword)
        # Skip whitespace
        while paren_idx < len(attrs_str) and attrs_str[paren_idx] == ' ':
            paren_idx += 1
        if paren_idx < len(attrs_str) and attrs_str[paren_idx] == '(':
            content, end = _extract_balanced_parens(attrs_str, paren_idx)
            results.append(content)
            i = end
        else:
            i = idx + len(keyword)
    return results


def _apply_attributes(attrs_str, arg):
    """Parse and apply attribute strings to an argument dict.

    Handles: intent(...), optional, dimension(...), depend(...), check(...)
    """
    if not attrs_str:
        return

    # Extract intent(...)
    for content in _find_keyword_args(attrs_str, 'intent'):
        parsed = _parse_intent(content)
        arg.setdefault('intents', []).extend(parsed['intents'])
        # Deduplicate
        arg['intents'] = list(dict.fromkeys(arg['intents']))
        if parsed['out_name']:
            arg['out_name'] = parsed['out_name']

    # Extract dimension(...)
    for content in _find_keyword_args(attrs_str, 'dimension'):
        arg['dimension'] = content.strip()

    # Extract depend(...)
    for content in _find_keyword_args(attrs_str, 'depend'):
        deps = [d.strip() for d in content.split(',')]
        arg.setdefault('depend', []).extend(deps)

    # Extract check(...)
    for content in _find_keyword_args(attrs_str, 'check'):
        arg.setdefault('checks', []).append(content)

    # Check for 'optional'
    if re.search(r'\boptional\b', attrs_str, re.I):
        arg['optional'] = True


def _extract_routine_blocks(pyf_text):
    """Extract individual routine blocks from expanded .pyf text.

    Returns a list of (block_text, block_type) tuples.
    """
    blocks = []

    # Find all subroutine/function blocks (not inside python module/interface)
    # Pattern: starts with "subroutine name(...)" or "[type] function name(...)"
    # ends with "end subroutine name" or "end function name"

    # We need to handle multi-line and various formatting
    pattern = re.compile(
        r'^[ \t]*(?:(\w[\w\s]*?)\s+)?'  # optional return type
        r'(subroutine|function)\s+'
        r'(\w+)\s*\([^)]*\)'  # name and args
        r'.*?'  # rest of first line (including result clause)
        r'(?=\n)',  # up to newline
        re.MULTILINE | re.IGNORECASE
    )

    end_pattern = re.compile(
        r'^[ \t]*end\s+(subroutine|function)\b.*$',
        re.MULTILINE | re.IGNORECASE
    )

    pos = 0
    while pos < len(pyf_text):
        m = pattern.search(pyf_text, pos)
        if m is None:
            break

        start = m.start()
        # Find the matching end
        end_m = end_pattern.search(pyf_text, m.end())
        if end_m is None:
            break

        block_text = pyf_text[start:end_m.end()]
        blocks.append(block_text)
        pos = end_m.end()

    return blocks


def parse_pyf_text(pyf_text):
    """Parse expanded .pyf text into a list of routine dicts.

    Parameters
    ----------
    pyf_text : str
        The fully expanded .pyf file content (after template expansion).

    Returns
    -------
    routines : list of dict
        Each dict describes one routine (subroutine or function).
    """
    blocks = _extract_routine_blocks(pyf_text)
    routines = []
    for block in blocks:
        routine = _parse_routine_block(block)
        if routine is not None:
            routines.append(routine)
    return routines


def parse_pyf_file(source_path):
    """Parse a .pyf.src file into a list of routine dicts.

    This expands templates first, then parses each routine.

    Parameters
    ----------
    source_path : str
        Path to a .pyf.src file.

    Returns
    -------
    routines : list of dict
        Each dict describes one routine.
    """
    expanded = expand_pyf_src(source_path)
    return parse_pyf_text(expanded)


# ---------------------------------------------------------------------------
# Convenience: dump parsed routines as JSON for inspection
# ---------------------------------------------------------------------------

def main():
    import json

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.pyf.src> [--json]")
        sys.exit(1)

    source = sys.argv[1]
    routines = parse_pyf_file(source)

    print(f"Parsed {len(routines)} routines from {source}")
    print()

    if '--json' in sys.argv:
        print(json.dumps(routines, indent=2))
    else:
        for r in routines:
            args_summary = []
            for aname in r['arg_names']:
                ainfo = r['args'].get(aname, {})
                intents = ainfo.get('intents', [])
                opt = 'opt' if ainfo.get('optional') else 'req'
                ftype = ainfo.get('ftype', '?')
                dim = ainfo.get('dimension', '')
                if dim:
                    dim = f'({dim})'
                args_summary.append(
                    f"  {aname}: {ftype}{dim} [{','.join(intents)}] {opt}"
                )
            hidden_args = []
            for aname, ainfo in r['args'].items():
                if aname not in r['arg_names']:
                    intents = ainfo.get('intents', [])
                    if 'hide' in intents:
                        default = ainfo.get('default', '?')
                        hidden_args.append(f"  {aname} = {default} (hidden)")

            print(f"{r['type']} {r['name']}:")
            if r['callstatement']:
                cs = r['callstatement']
                if len(cs) > 80:
                    cs = cs[:77] + '...'
                print(f"  callstatement: {cs}")
            for a in args_summary:
                print(a)
            for a in hidden_args:
                print(a)
            print()


if __name__ == '__main__':
    main()
