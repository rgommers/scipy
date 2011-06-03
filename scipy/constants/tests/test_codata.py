import warnings

import scipy.constants as constants
from scipy.constants import find, value
from numpy.testing import assert_equal, assert_, run_module_suite, \
                          assert_almost_equal

def test_find():

    warnings.simplefilter('ignore', DeprecationWarning)

    keys = find('weak mixing', disp=False)
    assert_equal(keys, ['weak mixing angle'])

    keys = find('qwertyuiop', disp=False)
    assert_equal(keys, [])

    keys = find('natural unit', disp=False)
    assert_equal(keys, sorted(['natural unit of velocity',
                                'natural unit of action',
                                'natural unit of action in eV s',
                                'natural unit of mass',
                                'natural unit of energy',
                                'natural unit of energy in MeV',
                                'natural unit of momentum',
                                'natural unit of momentum in MeV/c',
                                'natural unit of length',
                                'natural unit of time']))


def test_basic_table_parse():
    c = 'speed of light in vacuum'
    assert_equal(value(c), constants.c)
    assert_equal(value(c), constants.speed_of_light)

def test_basic_lookup():
    assert_equal('%d %s' % (constants.c, constants.unit('speed of light in vacuum')),
                 '299792458 m s^-1')

def test_find_all():
    assert_(len(find(disp=False)) > 300)

def test_find_single():
    assert_equal(find('Wien freq', disp=False)[0],
                 'Wien frequency displacement law constant')

def test_2002_vs_2006():
    assert_almost_equal(value('magn. flux quantum'),
                        value('mag. flux quantum'))

if __name__ == "__main__":
    run_module_suite()
