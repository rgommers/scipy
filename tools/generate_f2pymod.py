"""
Process f2py template files (`filename.pyf.src` -> `filename.pyf`)

Usage: python generate_pyf.py filename.pyf.src -o filename.pyf
"""

import os
import sys
import subprocess
import argparse

from numpy.distutils.from_template import process_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="Path to the input file")
    parser.add_argument("-o", "--outdir", type=str,
                        help="Path to the output directory")
    args = parser.parse_args()

    # Read .pyf.src file
    code = process_file(args.infile)

    # Write out the .pyf file
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    fname_pyf = os.path.join(args.outdir,
                             os.path.splitext(os.path.split(args.infile)[1])[0])

    with open(fname_pyf, 'w') as f:
        f.write(code)

    # Now invoke f2py to generate the C API module file
    p = subprocess.Popen([sys.executable, '-m', 'numpy.f2py', fname_pyf,
                          '--build-dir', outdir_abs], #'--quiet'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          cwd=os.getcwd())
    out, err = p.communicate()
    if not (p.returncode == 0):
        raise RuntimeError(f"Writing {args.outfile} with f2py failed!\n"
                           f"{out}\n"
                           r"{err}")


if __name__ == "__main__":
    main()
