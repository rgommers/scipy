#!/usr/bin/env python3
"""
refguide_check.py [OPTIONS] [-- ARGS]

Check for a Scipy submodule whether the objects in its __all__ dict
correspond to the objects included in the reference guide.

Example of usage::

    $ python3 refguide_check.py optimize

Note that this is a helper script to be able to check if things are missing;
the output of this script does need to be checked manually.  In some cases
objects are left out of the refguide for a good reason (it's an alias of
another function, or deprecated, or ...)

Another use of this helper script is to check validity of code samples
in docstrings. This is different from doctesting [we do not aim to have
scipy docstrings doctestable!], this is just to make sure that code in
docstrings is valid python::

    $ python3 refguide_check.py --doctests optimize

"""
import copy
import doctest
import glob
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager, redirect_stderr
from doctest import NORMALIZE_WHITESPACE, ELLIPSIS, IGNORE_EXCEPTION_DETAIL

import docutils.core
import numpy as np
from docutils.parsers.rst import directives

from numpydoc.docscrape_sphinx import get_doc_object
from numpydoc.docscrape import NumpyDocString
from scipy.stats._distr_params import distcont, distdiscrete
from scipy import stats


# Enable specific Sphinx directives
from sphinx.directives.other import SeeAlso, Only
directives.register_directive('seealso', SeeAlso)
directives.register_directive('only', Only)

BASE_MODULE = "scipy"

PUBLIC_SUBMODULES = [
    'cluster',
    'cluster.hierarchy',
    'cluster.vq',
    'constants',
    'datasets',
    'fft',
    'fftpack',
    'fftpack.convolve',
    'integrate',
    'interpolate',
    'io',
    'io.arff',
    'io.matlab',
    'io.wavfile',
    'linalg',
    'linalg.blas',
    'linalg.lapack',
    'linalg.interpolative',
    'misc',
    'ndimage',
    'odr',
    'optimize',
    'signal',
    'signal.windows',
    'sparse',
    'sparse.csgraph',
    'sparse.linalg',
    'spatial',
    'spatial.distance',
    'spatial.transform',
    'special',
    'stats',
    'stats.mstats',
    'stats.contingency',
    'stats.qmc',
    'stats.sampling'
]

# Docs for these modules are included in the parent module
OTHER_MODULE_DOCS = {
    'fftpack.convolve': 'fftpack',
    'io.wavfile': 'io',
    'io.arff': 'io',
}

# these names are known to fail doctesting and we like to keep it that way
# e.g. sometimes pseudocode is acceptable etc
DOCTEST_SKIPLIST = set([
    'scipy.stats.kstwobign',  # inaccurate cdf or ppf
    'scipy.stats.levy_stable',
    'scipy.special.sinc',  # comes from numpy
    'scipy.fft.fftfreq',
    'scipy.fft.rfftfreq',
    'scipy.fft.fftshift',
    'scipy.fft.ifftshift',
    'scipy.fftpack.fftfreq',
    'scipy.fftpack.fftshift',
    'scipy.fftpack.ifftshift',
    'scipy.integrate.trapezoid',
    'scipy.linalg.LinAlgError',
    'scipy.optimize.show_options',
    'io.rst',   # XXX: need to figure out how to deal w/ mat files
])

# these names are not required to be present in ALL despite being in
# autosummary:: listing
REFGUIDE_ALL_SKIPLIST = [
    r'scipy\.sparse\.csgraph',
    r'scipy\.sparse\.linalg',
    r'scipy\.linalg\.blas\.[sdczi].*',
    r'scipy\.linalg\.lapack\.[sdczi].*',
]

# these names are not required to be in an autosummary:: listing
# despite being in ALL
REFGUIDE_AUTOSUMMARY_SKIPLIST = [
    r'scipy\.special\..*_roots',  # old aliases for scipy.special.*_roots
    r'scipy\.special\.jn',  # alias for jv
    r'scipy\.ndimage\.sum',   # alias for sum_labels
    r'scipy\.integrate\.simps',   # alias for simpson
    r'scipy\.integrate\.trapz',   # alias for trapezoid
    r'scipy\.integrate\.cumtrapz',   # alias for cumulative_trapezoid
    r'scipy\.linalg\.solve_lyapunov',  # deprecated name
    r'scipy\.stats\.contingency\.chi2_contingency',
    r'scipy\.stats\.contingency\.expected_freq',
    r'scipy\.stats\.contingency\.margins',
    r'scipy\.stats\.reciprocal',  # alias for lognormal
    r'scipy\.stats\.trapz',   # alias for trapezoid
]
# deprecated windows in scipy.signal namespace
for name in ('barthann', 'bartlett', 'blackmanharris', 'blackman', 'bohman',
             'boxcar', 'chebwin', 'cosine', 'exponential', 'flattop',
             'gaussian', 'general_gaussian', 'hamming', 'hann', 'hanning',
             'kaiser', 'nuttall', 'parzen', 'triang', 'tukey'):
    REFGUIDE_AUTOSUMMARY_SKIPLIST.append(r'scipy\.signal\.' + name)

HAVE_MATPLOTLIB = False


def short_path(path, cwd=None):
    """
    Return relative or absolute path name, whichever is shortest.
    """
    if not isinstance(path, str):
        return path
    if cwd is None:
        cwd = os.getcwd()
    abspath = os.path.abspath(path)
    relpath = os.path.relpath(path, cwd)
    if len(abspath) <= len(relpath):
        return abspath
    return relpath


def find_names(module, names_dict):
    # Refguide entries:
    #
    # - 3 spaces followed by function name, and maybe some spaces, some
    #   dashes, and an explanation; only function names listed in
    #   refguide are formatted like this (mostly, there may be some false
    #   positives)
    #
    # - special directives, such as data and function
    #
    # - (scipy.constants only): quoted list
    #
    patterns = [
        r"^\s\s\s([a-z_0-9A-Z]+)(\s+-+.*)?$",
        r"^\.\. (?:data|function)::\s*([a-z_0-9A-Z]+)\s*$"
    ]

    if module.__name__ == 'scipy.constants':
        patterns += ["^``([a-z_0-9A-Z]+)``"]

    patterns = [re.compile(pattern) for pattern in patterns]
    module_name = module.__name__

    for line in module.__doc__.splitlines():
        res = re.search(
            r"^\s*\.\. (?:currentmodule|module):: ([a-z0-9A-Z_.]+)\s*$",
            line
        )
        if res:
            module_name = res.group(1)
            continue

        for pattern in patterns:
            res = re.match(pattern, line)
            if res is not None:
                name = res.group(1)
                names_dict.setdefault(module_name, set()).add(name)
                break


def get_all_dict(module):
    """Return a copy of the __all__ dict with irrelevant items removed."""
    if hasattr(module, "__all__"):
        all_dict = copy.deepcopy(module.__all__)
    else:
        all_dict = copy.deepcopy(dir(module))
        all_dict = [name for name in all_dict
                    if not name.startswith("_")]
    for name in ['absolute_import', 'division', 'print_function']:
        try:
            all_dict.remove(name)
        except ValueError:
            pass

    # Modules are almost always private; real submodules need a separate
    # run of refguide_check.
    all_dict = [name for name in all_dict
                if not inspect.ismodule(getattr(module, name, None))]

    deprecated = []
    not_deprecated = []
    for name in all_dict:
        f = getattr(module, name, None)
        if callable(f) and is_deprecated(f):
            deprecated.append(name)
        else:
            not_deprecated.append(name)

    others = (set(dir(module))
              .difference(set(deprecated))
              .difference(set(not_deprecated)))

    return not_deprecated, deprecated, others


def compare(all_dict, others, names, module_name):
    """Return sets of objects only in __all__, refguide, or completely missing."""
    only_all = set()
    for name in all_dict:
        if name not in names:
            for pat in REFGUIDE_AUTOSUMMARY_SKIPLIST:
                if re.match(pat, module_name + '.' + name):
                    break
            else:
                only_all.add(name)

    only_ref = set()
    missing = set()
    for name in names:
        if name not in all_dict:
            for pat in REFGUIDE_ALL_SKIPLIST:
                if re.match(pat, module_name + '.' + name):
                    if name not in others:
                        missing.add(name)
                    break
            else:
                only_ref.add(name)

    return only_all, only_ref, missing


def is_deprecated(f):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        try:
            f(**{"not a kwarg":None})
        except DeprecationWarning:
            return True
        except Exception:
            pass
        return False


def check_items(all_dict, names, deprecated, others, module_name, dots=True):
    num_all = len(all_dict)
    num_ref = len(names)

    output = ""

    output += "Non-deprecated objects in __all__: %i\n" % num_all
    output += "Objects in refguide: %i\n\n" % num_ref

    only_all, only_ref, missing = compare(all_dict, others, names, module_name)
    dep_in_ref = only_ref.intersection(deprecated)
    only_ref = only_ref.difference(deprecated)

    if len(dep_in_ref) > 0:
        output += "Deprecated objects in refguide::\n\n"
        for name in sorted(deprecated):
            output += "    " + name + "\n"

    if len(only_all) == len(only_ref) == len(missing) == 0:
        if dots:
            output_dot('.')
        return [(None, True, output)]
    else:
        if len(only_all) > 0:
            output += (
                f"ERROR: objects in {module_name}.__all__ but not in refguide::\n\n"
            )
            for name in sorted(only_all):
                output += "    " + name + "\n"

            output += "\nThis issue can be fixed by adding these objects to\n"
            output += "the function listing in __init__.py for this module\n"

        if len(only_ref) > 0:
            output += (
                f"ERROR: objects in refguide but not in {module_name}.__all__::\n\n"
            )
            for name in sorted(only_ref):
                output += "    " + name + "\n"

            output += "\nThis issue should likely be fixed by removing these objects\n"
            output += "from the function listing in __init__.py for this module\n"
            output += "or adding them to __all__.\n"

        if len(missing) > 0:
            output += "ERROR: missing objects::\n\n"
            for name in sorted(missing):
                output += "    " + name + "\n"

        if dots:
            output_dot('F')
        return [(None, False, output)]


def validate_rst_syntax(text, name, dots=True):
    if text is None:
        if dots:
            output_dot('E')
        return False, f"ERROR: {name}: no documentation"

    ok_unknown_items = set([
        'mod', 'currentmodule', 'autosummary', 'data', 'legacy',
        'obj', 'versionadded', 'versionchanged', 'module', 'class', 'meth',
        'ref', 'func', 'toctree', 'moduleauthor', 'deprecated',
        'sectionauthor', 'codeauthor', 'eq', 'doi', 'DOI', 'arXiv', 'arxiv'
    ])

    # Run through docutils
    error_stream = io.StringIO()

    def resolve(name, is_label=False):
        return ("http://foo", name)

    token = '<RST-VALIDATE-SYNTAX-CHECK>'

    docutils.core.publish_doctree(
        text, token,
        settings_overrides = dict(halt_level=5,
                                  traceback=True,
                                  default_reference_context='title-reference',
                                  default_role='emphasis',
                                  link_base='',
                                  resolve_name=resolve,
                                  stylesheet_path='',
                                  raw_enabled=0,
                                  file_insertion_enabled=0,
                                  warning_stream=error_stream))

    # Print errors, disregarding unimportant ones
    error_msg = error_stream.getvalue()
    errors = error_msg.split(token)
    success = True
    output = ""

    for error in errors:
        lines = error.splitlines()
        if not lines:
            continue

        m = re.match(
            r'.*Unknown (?:interpreted text role|directive type) "(.*)".*$',
            lines[0]
        )
        if m:
            if m.group(1) in ok_unknown_items:
                continue

        m = re.match(
            r'.*Error in "math" directive:.*unknown option: "label"', " ".join(lines),
            re.S
        )
        if m:
            continue

        output += (
            name + lines[0] + "::\n    " + "\n    ".join(lines[1:]).rstrip() + "\n"
        )
        success = False

    if not success:
        output += "    " + "-"*72 + "\n"
        for lineno, line in enumerate(text.splitlines()):
            output += "    %-4d    %s\n" % (lineno+1, line)
        output += "    " + "-"*72 + "\n\n"

    if dots:
        output_dot('.' if success else 'F')
    return success, output


def output_dot(msg='.', stream=sys.stderr):
    stream.write(msg)
    stream.flush()


def check_rest(module, names, dots=True):
    """
    Check reStructuredText formatting of docstrings

    Returns: [(name, success_flag, output), ...]
    """
    skip_types = (dict, str, float, int)

    results = []

    if module.__name__[6:] not in OTHER_MODULE_DOCS:
        results += [(module.__name__,) +
                    validate_rst_syntax(inspect.getdoc(module),
                                        module.__name__, dots=dots)]

    for name in names:
        full_name = module.__name__ + '.' + name
        obj = getattr(module, name, None)

        if obj is None:
            results.append((full_name, False, f"{full_name} has no docstring"))
            continue
        elif isinstance(obj, skip_types):
            continue

        if inspect.ismodule(obj):
            text = inspect.getdoc(obj)
        else:
            try:
                text = str(get_doc_object(obj))
            except Exception:
                import traceback
                results.append((full_name, False,
                                "Error in docstring format!\n" +
                                traceback.format_exc()))
                continue

        m = re.search(".*?([\x00-\x09\x0b-\x1f]).*", text)
        if m:
            msg = ("Docstring contains a non-printable character "
                   f"{m.group(1)!r} in the line\n\n{m.group(0)!r}\n\n"
                   "Maybe forgot r\"\"\"?")
            results.append((full_name, False, msg))
            continue

        try:
            src_file = short_path(inspect.getsourcefile(obj))
        except TypeError:
            src_file = None

        if src_file:
            file_full_name = src_file + ':' + full_name
        else:
            file_full_name = full_name

        results.append(
            (full_name,) + validate_rst_syntax(text, file_full_name, dots=dots)
        )

    return results


### Doctest helpers ####

# the namespace to run examples in
DEFAULT_NAMESPACE = {}

# the namespace to do checks in
CHECK_NAMESPACE = {
      'np': np,
      'assert_allclose': np.testing.assert_allclose,
      'assert_equal': np.testing.assert_equal,
      # recognize numpy repr's
      'array': np.array,
      'matrix': np.matrix,
      'masked_array': np.ma.masked_array,
      'int64': np.int64,
      'uint64': np.uint64,
      'int8': np.int8,
      'int32': np.int32,
      'float32': np.float32,
      'float64': np.float64,
      'dtype': np.dtype,
      'nan': np.nan,
      'NaN': np.nan,
      'inf': np.inf,
      'Inf': np.inf,}


def try_convert_namedtuple(got):
    # suppose that "got" is smth like MoodResult(statistic=10, pvalue=0.1).
    # Then convert it to the tuple (10, 0.1), so that can later compare tuples.
    num = got.count('=')
    if num == 0:
        # not a nameduple, bail out
        return got
    regex = (r'[\w\d_]+\(' +
             ', '.join([r'[\w\d_]+=(.+)']*num) +
             r'\)')
    grp = re.findall(regex, " ".join(got.split()))
    # fold it back to a tuple
    got_again = '(' + ', '.join(grp[0]) + ')'
    return got_again


class DTRunner(doctest.DocTestRunner):
    DIVIDER = "\n"

    def __init__(self, item_name, checker=None, verbose=None, optionflags=0):
        self._item_name = item_name
        self._had_unexpected_error = False
        doctest.DocTestRunner.__init__(self, checker=checker, verbose=verbose,
                                       optionflags=optionflags)

    def _report_item_name(self, out, new_line=False):
        if self._item_name is not None:
            if new_line:
                out("\n")
            self._item_name = None

    def report_start(self, out, test, example):
        self._checker._source = example.source
        return doctest.DocTestRunner.report_start(self, out, test, example)

    def report_success(self, out, test, example, got):
        if self._verbose:
            self._report_item_name(out, new_line=True)
        return doctest.DocTestRunner.report_success(self, out, test, example, got)

    def report_unexpected_exception(self, out, test, example, exc_info):
        # Ignore name errors after failing due to an unexpected exception
        exception_type = exc_info[0]
        if self._had_unexpected_error and exception_type is NameError:
            return
        self._had_unexpected_error = True

        self._report_item_name(out)
        return super().report_unexpected_exception(
            out, test, example, exc_info)

    def report_failure(self, out, test, example, got):
        self._report_item_name(out)
        return doctest.DocTestRunner.report_failure(self, out, test,
                                                    example, got)


class Checker(doctest.OutputChecker):
    obj_pattern = re.compile(r'at 0x[0-9a-fA-F]+>')
    vanilla = doctest.OutputChecker()
    rndm_markers = {'# random', '# Random', '#random', '#Random', "# may vary"}
    stopwords = {'plt.', '.hist', '.show', '.ylim', '.subplot(',
                 'set_title', 'imshow', 'plt.show', '.axis(', '.plot(',
                 '.bar(', '.title', '.ylabel', '.xlabel', 'set_ylim', 'set_xlim',
                 '# reformatted', '.set_xlabel(', '.set_ylabel(', '.set_zlabel(',
                 '.set(xlim=', '.set(ylim=', '.set(xlabel=', '.set(ylabel='}

    def __init__(self, parse_namedtuples=True, ns=None, atol=1e-8, rtol=1e-2):
        self.parse_namedtuples = parse_namedtuples
        self.atol, self.rtol = atol, rtol
        if ns is None:
            self.ns = dict(CHECK_NAMESPACE)
        else:
            self.ns = ns

    def check_output(self, want, got, optionflags):
        # cut it short if they are equal
        if want == got:
            return True

        # skip stopwords in source
        if any(word in self._source for word in self.stopwords):
            return True

        # skip random stuff
        if any(word in want for word in self.rndm_markers):
            return True

        # skip function/object addresses
        if self.obj_pattern.search(got):
            return True

        # ignore comments (e.g. signal.freqresp)
        if want.lstrip().startswith("#"):
            return True

        # try the standard doctest
        try:
            if self.vanilla.check_output(want, got, optionflags):
                return True
        except Exception:
            pass

        # OK then, convert strings to objects
        try:
            a_want = eval(want, dict(self.ns))
            a_got = eval(got, dict(self.ns))
        except Exception:
            # Maybe we're printing a numpy array? This produces invalid python
            # code: `print(np.arange(3))` produces "[0 1 2]" w/o commas between
            # values. So, reinsert commas and retry.
            # TODO: handle (1) abbreviation (`print(np.arange(10000))`), and
            #              (2) n-dim arrays with n > 1
            s_want = want.strip()
            s_got = got.strip()
            cond = (s_want.startswith("[") and s_want.endswith("]") and
                    s_got.startswith("[") and s_got.endswith("]"))
            if cond:
                s_want = ", ".join(s_want[1:-1].split())
                s_got = ", ".join(s_got[1:-1].split())
                return self.check_output(s_want, s_got, optionflags)

            # maybe we are dealing with masked arrays?
            # their repr uses '--' for masked values and this is invalid syntax
            # If so, replace '--' by nans (they are masked anyway) and retry
            if 'masked_array' in want or 'masked_array' in got:
                s_want = want.replace('--', 'nan')
                s_got = got.replace('--', 'nan')
                return self.check_output(s_want, s_got, optionflags)

            if "=" not in want and "=" not in got:
                # if we're here, want and got cannot be eval-ed (hence cannot
                # be converted to numpy objects), they are not namedtuples
                # (those must have at least one '=' sign).
                # Thus they should have compared equal with vanilla doctest.
                # Since they did not, it's an error.
                return False

            if not self.parse_namedtuples:
                return False
            # suppose that "want"  is a tuple, and "got" is smth like
            # MoodResult(statistic=10, pvalue=0.1).
            # Then convert the latter to the tuple (10, 0.1),
            # and then compare the tuples.
            try:
                got_again = try_convert_namedtuple(got)
                want_again = try_convert_namedtuple(want)
            except Exception:
                return False
            else:
                return self.check_output(want_again, got_again, optionflags)

        # ... and defer to numpy
        try:
            return self._do_check(a_want, a_got)
        except Exception:
            # heterog tuple, eg (1, np.array([1., 2.]))
            try:
                return all(self._do_check(w, g) for w, g in zip(a_want, a_got))
            except (TypeError, ValueError):
                return False

    def _do_check(self, want, got):
        # This should be done exactly as written to correctly handle all of
        # numpy-comparable objects, strings, and heterogeneous tuples
        try:
            if want == got:
                return True
        except Exception:
            pass
        return np.allclose(want, got, atol=self.atol, rtol=self.rtol)


def _run_doctests(tests, full_name, verbose, doctest_warnings):
    """Run modified doctests for the set of `tests`.

    Returns: list of [(success_flag, output), ...]
    """
    flags = NORMALIZE_WHITESPACE | ELLIPSIS | IGNORE_EXCEPTION_DETAIL
    runner = DTRunner(full_name, checker=Checker(), optionflags=flags,
                      verbose=verbose)

    output = io.StringIO(newline='')
    success = True
    # Redirect stderr to the stdout or output
    tmp_stderr = sys.stdout if doctest_warnings else output
    from scipy._lib._util import _fixed_default_rng

    @contextmanager
    def temp_cwd():
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmpdir)

    # Run tests, trying to restore global state afterward
    cwd = os.getcwd()
    with np.errstate(), np.printoptions(), temp_cwd(), \
            redirect_stderr(tmp_stderr), \
            _fixed_default_rng():
        # try to ensure random seed is NOT reproducible
        np.random.seed(None)

        for t in tests:
            t.filename = short_path(t.filename, cwd)
            fails, successes = runner.run(t, out=output.write)
            if fails > 0:
                success = False

    output.seek(0)
    return success, output.read()


def check_doctests(module, verbose, ns=None,
                   dots=True, doctest_warnings=False):
    """Check code in docstrings of the module's public symbols.

    Returns: list of [(item_name, success_flag, output), ...]
    """
    if ns is None:
        ns = dict(DEFAULT_NAMESPACE)

    # Loop over non-deprecated items
    results = []

    for name in get_all_dict(module)[0]:
        full_name = module.__name__ + '.' + name

        if full_name in DOCTEST_SKIPLIST:
            continue

        try:
            obj = getattr(module, name)
        except AttributeError:
            import traceback
            results.append((full_name, False,
                            "Missing item!\n" +
                            traceback.format_exc()))
            continue

        finder = doctest.DocTestFinder()
        try:
            tests = finder.find(obj, name, globs=dict(ns))
        except Exception:
            import traceback
            results.append((full_name, False,
                            "Failed to get doctests!\n" +
                            traceback.format_exc()))
            continue

        success, output = _run_doctests(tests, full_name, verbose,
                                        doctest_warnings)

        if dots:
            output_dot('.' if success else 'F')

        results.append((full_name, success, output))

        if HAVE_MATPLOTLIB:
            import matplotlib.pyplot as plt
            plt.close('all')

    return results


def check_doctests_testfile(fname, verbose, ns=None,
                   dots=True, doctest_warnings=False):
    """Check code in a text file.

    Mimic `check_doctests` above, differing mostly in test discovery.
    (which is borrowed from stdlib's doctest.testfile here,
     https://github.com/python-git/python/blob/master/Lib/doctest.py)

    Returns: list of [(item_name, success_flag, output), ...]

    Notes
    -----

    refguide can be signalled to skip testing code by adding
    ``#doctest: +SKIP`` to the end of the line. If the output varies or is
    random, add ``# may vary`` or ``# random`` to the comment. for example

    >>> plt.plot(...)  # doctest: +SKIP
    >>> random.randint(0,10)
    5 # random

    We also try to weed out pseudocode:
    * We maintain a list of exceptions which signal pseudocode,
    * We split the text file into "blocks" of code separated by empty lines
      and/or intervening text.
    * If a block contains a marker, the whole block is then assumed to be
      pseudocode. It is then not being doctested.

    The rationale is that typically, the text looks like this:

    blah
    <BLANKLINE>
    >>> from numpy import some_module   # pseudocode!
    >>> func = some_module.some_function
    >>> func(42)                  # still pseudocode
    146
    <BLANKLINE>
    blah
    <BLANKLINE>
    >>> 2 + 3        # real code, doctest it
    5

    """
    results = []

    if ns is None:
        ns = dict(DEFAULT_NAMESPACE)

    _, short_name = os.path.split(fname)
    if short_name in DOCTEST_SKIPLIST:
        return results

    full_name = fname
    with open(fname, encoding='utf-8') as f:
        text = f.read()

    PSEUDOCODE = set(['some_function', 'some_module', 'import example',
                      'ctypes.CDLL',     # likely need compiling, skip it
                      'integrate.nquad(func,'  # ctypes integrate tutotial
                      ])

    # split the text into "blocks" and try to detect and omit pseudocode blocks.
    parser = doctest.DocTestParser()
    good_parts = []
    for part in text.split('\n\n'):
        tests = parser.get_doctest(part, ns, fname, fname, 0)
        if any(word in ex.source for word in PSEUDOCODE
                                 for ex in tests.examples):
            # omit it
            pass
        else:
            # `part` looks like a good code, let's doctest it
            good_parts += [part]

    # Reassemble the good bits and doctest them:
    good_text = '\n\n'.join(good_parts)
    tests = parser.get_doctest(good_text, ns, fname, fname, 0)
    success, output = _run_doctests([tests], full_name, verbose,
                                    doctest_warnings)

    if dots:
        output_dot('.' if success else 'F')

    results.append((full_name, success, output))

    if HAVE_MATPLOTLIB:
        import matplotlib.pyplot as plt
        plt.close('all')

    return results


def init_matplotlib():
    global HAVE_MATPLOTLIB

    try:
        import matplotlib
        matplotlib.use('Agg')
        HAVE_MATPLOTLIB = True
    except ImportError:
        HAVE_MATPLOTLIB = False


def check_dist_keyword_names():
    # Look for collisions between names of distribution shape parameters and
    # keywords of distribution methods. See gh-5982.
    distnames = set(distdata[0] for distdata in distcont + distdiscrete)
    mod_results = []
    for distname in distnames:
        dist = getattr(stats, distname)

        method_members = inspect.getmembers(dist, predicate=inspect.ismethod)
        method_names = [method[0] for method in method_members
                        if not method[0].startswith('_')]
        for methodname in method_names:
            method = getattr(dist, methodname)
            try:
                params = NumpyDocString(method.__doc__)['Parameters']
            except TypeError:
                result = (f'stats.{distname}.{methodname}', False,
                          "Method parameters are not documented properly.")
                mod_results.append(result)
                continue

            if not dist.shapes:  # can't have collision if there are no shapes
                continue
            shape_names = dist.shapes.split(', ')

            param_names1 = set(param.name for param in params)
            param_names2 = set(inspect.signature(method).parameters)
            param_names = param_names1.union(param_names2)

            # # Disabling this check in this PR;
            # # these discrepancies are a separate issue.
            # no_doc_params = {'args', 'kwds', 'kwargs'}  # no need to document
            # undoc_params = param_names2 - param_names1 - no_doc_params
            # if un_doc_params:
            #     result = (f'stats.{distname}.{methodname}', False,
            #               f'Parameter(s) {undoc_params} are not documented.')
            #     mod_results.append(result)
            #     continue

            intersection = param_names.intersection(shape_names)

            if intersection:
                message = ("Distribution/method keyword collision: "
                           f"{intersection} ")
                result = (f'stats.{distname}.{methodname}', False, message)
            else:
                result = (f'stats.{distname}.{methodname}', True, '')
            mod_results.append(result)

    return mod_results


def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("module_names", metavar="SUBMODULES", default=[],
                        nargs='*', help="Submodules to check (default: all public)")
    parser.add_argument("--doctests", action="store_true", help="Run also doctests")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--doctest-warnings", action="store_true",
                        help="Enforce warning checking for doctests")
    parser.add_argument("--skip-tutorial", action="store_true",
                        help="Skip running doctests in the tutorial.")
    args = parser.parse_args(argv)

    modules = []
    names_dict = {}

    if args.module_names:
        args.skip_tutorial = True
    else:
        args.module_names = list(PUBLIC_SUBMODULES)

    os.environ['SCIPY_PIL_IMAGE_VIEWER'] = 'true'

    module_names = list(args.module_names)
    for name in list(module_names):
        if name in OTHER_MODULE_DOCS:
            name = OTHER_MODULE_DOCS[name]
            if name not in module_names:
                module_names.append(name)

    for submodule_name in module_names:
        prefix = BASE_MODULE + '.'
        if not submodule_name.startswith(prefix):
            module_name = prefix + submodule_name
        else:
            module_name = submodule_name

        __import__(module_name)
        module = sys.modules[module_name]

        if submodule_name not in OTHER_MODULE_DOCS:
            find_names(module, names_dict)

        if submodule_name in args.module_names:
            modules.append(module)

    dots = True
    success = True
    results = []

    print("Running checks for %d modules:" % (len(modules),))

    if args.doctests or not args.skip_tutorial:
        init_matplotlib()

    for module in modules:
        if dots:
            if module is not modules[0]:
                sys.stderr.write(' ')
            sys.stderr.write(module.__name__ + ' ')
            sys.stderr.flush()

        all_dict, deprecated, others = get_all_dict(module)
        names = names_dict.get(module.__name__, set())

        mod_results = []
        mod_results += check_items(all_dict, names, deprecated, others, module.__name__)
        mod_results += check_rest(module, set(names).difference(deprecated),
                                  dots=dots)
        if args.doctests:
            mod_results += check_doctests(module, (args.verbose >= 2), dots=dots,
                                          doctest_warnings=args.doctest_warnings)
        if module.__name__ == 'scipy.stats':
            mod_results += check_dist_keyword_names()

        for v in mod_results:
            assert isinstance(v, tuple), v

        results.append((module, mod_results))

    if dots:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if not args.skip_tutorial:
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        tut_path = os.path.join(base_dir, 'doc', 'source', 'tutorial', '*.rst')
        print(f'\nChecking tutorial files at {os.path.relpath(tut_path, os.getcwd())}:')
        for filename in sorted(glob.glob(tut_path)):
            if dots:
                sys.stderr.write('\n')
                sys.stderr.write(os.path.split(filename)[1] + ' ')
                sys.stderr.flush()

            tut_results = check_doctests_testfile(filename, (args.verbose >= 2),
                    dots=dots, doctest_warnings=args.doctest_warnings)

            def scratch():
                pass        # stub out a "module", see below
            scratch.__name__ = filename
            results.append((scratch, tut_results))

        if dots:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # Report results
    all_success = True

    for module, mod_results in results:
        success = all(x[1] for x in mod_results)
        all_success = all_success and success

        if success and args.verbose == 0:
            continue

        print("")
        print("=" * len(module.__name__))
        print(module.__name__)
        print("=" * len(module.__name__))
        print("")

        for name, success, output in mod_results:
            if name is None:
                if not success or args.verbose >= 1:
                    print(output.strip())
                    print("")
            elif not success or (args.verbose >= 2 and output.strip()):
                print(name)
                print("-"*len(name))
                print("")
                print(output.strip())
                print("")

    if all_success:
        print("\nOK: refguide and doctests checks passed!")
        sys.exit(0)
    else:
        print("\nERROR: refguide or doctests have errors")
        sys.exit(1)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
