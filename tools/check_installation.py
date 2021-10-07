import os
import glob
import sys

CUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(CUR_DIR)
SCIPY_DIR = os.path.join(ROOT_DIR, 'scipy')

def main(argv):
    INSTALLED_DIR = os.path.join(ROOT_DIR, argv[0])
    scipy_test_files = get_test_files(SCIPY_DIR)
    installed_test_files = get_test_files(INSTALLED_DIR)
    for test_file in scipy_test_files.keys():
        if not test_file in installed_test_files.keys():
            print("%s is not installed" % scipy_test_files[test_file])
            #raise
    print("-"*30)
    for test_file in installed_test_files.keys():
        if not test_file in scipy_test_files.keys():
            print("%s is installed at improper location" % installed_test_files[test_file])
            #raise
    #print("All the test files were installed")

def get_parent_dir(current_path, levels = 1):
    current_new = current_path
    for i in range(levels + 1):
        current_new = os.path.dirname(current_new)
    return os.path.relpath(current_path, current_new)

def get_test_files(dir):
    test_files = dict()
    for path in glob.glob(f'{dir}/**/test_*.py', recursive=True):
        test_files[get_parent_dir(path, 3)] = path
    return test_files

if __name__ == '__main__':
    main(argv=sys.argv[1:])
