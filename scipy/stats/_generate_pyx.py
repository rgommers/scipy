import pathlib
from shutil import copyfile
import subprocess
import sys
import os
import argparse


def isNPY_OLD():
    '''
    A new random C API was added in 1.18 and became stable in 1.19.
    Prefer the new random C API when building with recent numpy.
    '''
    import numpy as np
    ver = tuple(int(num) for num in np.__version__.split('.')[:2])
    return ver < (1, 19)


def make_biasedurn(outdir):
    '''Substitute True/False values for NPY_OLD Cython build variable.'''
    biasedurn_base = (pathlib.Path(__file__).parent / 'biasedurn').absolute()
    with open(biasedurn_base.with_suffix('.pyx.templ'), 'r') as src:
        contents = src.read()

    outfile = outdir / 'biasedurn.pyx'
    with open(outfile, 'w') as dest:
        dest.write(contents.format(NPY_OLD=str(bool(isNPY_OLD()))))


def make_boost(outdir, distutils_build=False):
    # Call code generator inside _boost directory
    code_gen = pathlib.Path(__file__).parent / '_boost/include/code_gen.py'
    if distutils_build:
        subprocess.run([sys.executable, str(code_gen), '-o', outdir,
                        '--distutils-build', 'True'], check=True)
    else:
        subprocess.run([sys.executable, str(code_gen), '-o', outdir],
                       check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", type=str,
                        help="Path to the output directory")
    args = parser.parse_args()

    if not args.outdir:
        #raise ValueError(f"Missing `--outdir` argument to _generate_pyx.py")
        # We're dealing with a distutils build here, write in-place:
        outdir_abs = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
        make_biasedurn(outdir_abs)

        outdir_abs_boost = outdir_abs / '_boost' / 'src'
        if not os.path.exists(outdir_abs_boost):
            os.makedirs(outdir_abs_boost)
        make_boost(outdir_abs_boost, distutils_build=True)
    else:
        outdir_abs = pathlib.Path(os.getcwd()) / args.outdir
        make_biasedurn(outdir_abs)
        make_boost(outdir_abs)
