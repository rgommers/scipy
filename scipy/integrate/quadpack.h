/* MULTIPACK module by Travis Oliphant

Copyright (c) 1999 Travis Oliphant all rights reserved
oliphant.travis@ieee.org
Permission to use, modify, and distribute this software is given under the 
terms of the Scipy License

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
*/


/* This extension module is a collection of wrapper functions around
common FORTRAN code in the packages MINPACK, ODEPACK, and QUADPACK plus
some differential algebraic equation solvers.

The wrappers are meant to be nearly direct translations between the
FORTAN code and Python.  Some parameters like sizes do not need to be 
passed since they are available from the objects.  

It is anticipated that a pure Python module be written to call these lower
level routines and make a simpler user interface.  All of the routines define
default values for little-used parameters so that even the raw routines are
quite useful without a separate wrapper. 

FORTRAN Outputs that are not either an error indicator or the sought-after
results are placed in a dictionary and returned as an optional member of
the result tuple when the full_output argument is non-zero.
*/


#include "Python.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/arrayobject.h"
#include <setjmp.h>


#define PYERR(errobj,message) {PyErr_SetString(errobj,message); goto fail;}
#define PYERR2(errobj,message) {PyErr_Print(); PyErr_SetString(errobj, message); goto fail;}
#define ISCONTIGUOUS(m) ((m)->flags & CONTIGUOUS)


static PyObject *quadpack_python_function=NULL;
static PyObject *quadpack_extra_arguments=NULL;    /* a tuple */
static jmp_buf quadpack_jmpbuf;

static double (*quadpack_ctypes_function)(double) = NULL;

static PyObject *quadpack_error;

static double* globalargs; //Array to store function parameters (x[1],...,x[n])
static double (*globalf)(int, double *); //Pointer to function of array
static int globalnargs; //Int to store number of elements in globalargs
static double (*globalbasef)(double *); //Function received from __quadpack.h to initialize and convert to form used in wrapper

typedef struct { //Similar to QStorage: allows reentrancy
    //Last
    double *z_args0;
    int z_nargs0;
    double (*z_f0)(int, double *);
    //Current
    double *z_args1;
    int z_nargs1;
    double (*z_f1)(int, double *);
} ZStorage;

typedef struct { 
    //Last
    double (*y_func0)(double *);
    //Current
    double (*y_func1)(double *);
} YStorage;

/* Stack Storage for re-entrant capability */
typedef struct {
    void *global0;
    void *global1;
    jmp_buf jmp;    
    PyObject *arg;
} QStorage;

typedef double (*_sp_double_func)(double);

typedef struct {
    PyObject_HEAD
    char *b_ptr;
} _sp_cfuncptr_object;

static _sp_double_func
get_ctypes_function_pointer(PyObject *obj) {
    return (*((void **)(((_sp_cfuncptr_object *)(obj))->b_ptr)));
}

static int 
quad_init_func(QStorage *store, PyObject *fun, PyObject *arg) {
    store->global0 = (void *)quadpack_python_function;
    store->global1 = (void *)quadpack_extra_arguments;
    memcpy(&(store->jmp), &quadpack_jmpbuf, sizeof(jmp_buf));
    store->arg = arg;
    if (store->arg == NULL) {
        if ((store->arg = PyTuple_New(0)) == NULL) 
            return NPY_FAIL;
    }
    else {
        Py_INCREF(store->arg);  /* We decrement on restore */
    }
    if (!PyTuple_Check(store->arg)) {
        PyErr_SetString(quadpack_error, "Extra Arguments must be in a tuple");
        Py_XDECREF(store->arg);
        return NPY_FAIL;
    }
    quadpack_python_function = fun;
    quadpack_extra_arguments = store->arg;
    return NPY_SUCCEED;
}

static void
quad_restore_func(QStorage *store, int *ierr) {
    quadpack_python_function = (PyObject *)store->global0;
    quadpack_extra_arguments = (PyObject *)store->global1;
    memcpy(&quadpack_jmpbuf, &(store->jmp), sizeof(jmp_buf));
    Py_XDECREF(store->arg);
    if (ierr != NULL) {
        if (PyErr_Occurred()) {
            *ierr = 80;             /* Python error */
            PyErr_Clear();
        }
    }
}

static int
init_ctypes_func(QStorage *store, PyObject *fun) {
    store->global0 = quadpack_ctypes_function;
    store->global1 = get_ctypes_function_pointer(fun);
    if (store->global1 == NULL) return NPY_FAIL;
    quadpack_ctypes_function = store->global1;
    return NPY_SUCCEED;
}

static void
restore_ctypes_func(QStorage *store) {
    quadpack_ctypes_function = store->global0;
}

int init_c_multivariate(ZStorage* store, double (*f)(int, double *), int n, double args[n]){
  /*Initialize function of n+1 variables
  Input: 
    f - Function pointer to function to evaluate
    n - integer number of extra parameters 
    args - double array of length n with parameters x[1]....x[n]
  Output:
    NPY_FAIL on failure 
    NPY_SUCCEED on success
  */

  //Store current parameters
  store->z_f0 = globalf;      
  store->z_nargs0 = globalnargs;
  store->z_args0 = globalargs;

  //Store new parameters
  store->z_f1 = f;      
  store->z_nargs1 = n;
  store->z_args1 = args;
  if (store->z_f1 == NULL) return NPY_FAIL;
  
  //Set globals
  globalf = store->z_f1;
  globalnargs = store->z_nargs1;
  globalargs = store->z_args1;
  return NPY_SUCCEED;
}

double call_c_multivariate(double* x){ 
  /*Evaluates user defined function as function of one variable. 
    MUST BE INITIALIZED FIRST
  Input: Pointer to double x to evaluate function at
  Output: Function evaluated at x with initialized parameters
  We want to create a new array with [x0, concatenated with [x1, . . . , xn]]
  */ 
  double evalArray[globalnargs+1];
  int i = 1;
  evalArray[0] = *x;
  for(i; i < globalnargs + 1 ; i++){
    evalArray[i] = globalargs[i-1]; //Add everything from globalargs to end of evalArray
  }
  return globalf(globalnargs, evalArray);
}

void restore_c_multivariate(ZStorage* store){
  globalf = store->z_f0;
  globalnargs = store->z_nargs0;
  globalargs = store->z_args0;
  return;
}



/*Second wrapper. Interprets funciton of f(x) as f(n,x[n]) for use with
above cwrapper
For use: call funcwrapper_init(f(x))
         call routine2(funcwrapper, ...) (from above)
*/

void funcwrapper_init(YStorage* store, double (*f)(double *)){
  //sets f as global for future use
  //input: f - function of double pointer
  store->y_func0 = globalbasef;
  store->y_func1 = f;


  globalbasef = store->y_func1;
  return;
}

double funcwrapper(int nargs, double args[nargs]){
  /*Take globalbasef and evaluate it in the form that cwrapper
  can handle
  NOTE: This will need to be more complex to add support for multivariate functions,
  as currently only single variable functions are called through this wrapper.*/
  return globalbasef(args);  
}

void funcwrapper_restore(YStorage* store){
    //Restores function after use
    globalbasef = store->y_func0;
}