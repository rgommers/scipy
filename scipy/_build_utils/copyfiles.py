import os
from shutil import copyfile
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles", nargs='+',
                        help="Paths to the input files")
    parser.add_argument("-o", "--outdir", type=str,
                        help="Path to the output directory")
    args = parser.parse_args()

    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    for infile in args.infiles:
        outfile = os.path.join(outdir_abs, infile)
        copyfile(infile, outfile)


if __name__ == "__main__":
    main()
