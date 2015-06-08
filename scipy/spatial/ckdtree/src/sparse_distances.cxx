
#include <Python.h>
#include "numpy/arrayobject.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>
#include <sstream>
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>

#define CKDTREE_METHODS_IMPL
#include "ckdtree_decl.h"
#include "ckdtree_methods.h"
#include "cpp_exc.h"
#include "rectangle.h"
#include "coo_entries.h"

static void
traverse(const ckdtree *self, const ckdtree *other, 
         std::vector<coo_entry> *results,
         const ckdtreenode *node1, const ckdtreenode *node2,
         RectRectDistanceTracker *tracker)
{
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    
    npy_float64 d;
    npy_intp i, j, min_j;
            
    if (tracker->min_distance > tracker->upper_bound)
        return;
    else if (node1->split_dim == -1) {  /* 1 is leaf node */
        lnode1 = node1;
        
        if (node2->split_dim == -1) {  /* 1 & 2 are leaves */
            lnode2 = node2;
            
            /* brute-force */
            const npy_float64 p = tracker->p;
            const npy_float64 tub = tracker->upper_bound;
            const npy_float64 *sdata = self->raw_data;
            const npy_intp *sindices = self->raw_indices;
            const npy_float64 *odata = other->raw_data;
            const npy_intp *oindices = other->raw_indices;
            const npy_intp m = self->m;
            const npy_intp start1 = lnode1->start_idx;
            const npy_intp start2 = lnode2->start_idx;
            const npy_intp end1 = lnode1->end_idx;
            const npy_intp end2 = lnode2->end_idx;
                        
            prefetch_datapoint(sdata + sindices[start1] * m, m);
            if (start1 < end1)
               prefetch_datapoint(sdata + sindices[start1+1] * m, m);                         
                        
            for (i = start1; i < end1; ++i) {
            
                if (i < end1-2)
                     prefetch_datapoint(sdata + sindices[i+2] * m, m);
            
                /* Special care here to avoid duplicate pairs */
                if (node1 == node2)
                    min_j = i+1;
                else
                    min_j = start2;
                    
                prefetch_datapoint(odata + oindices[min_j] * m, m);
                if (min_j < end2)
                    prefetch_datapoint(sdata + oindices[min_j+1] * m, m);
                    
                for (j = min_j; j < end2; ++j) {
                
                    if (j < end2-2)
                        prefetch_datapoint(odata + oindices[j+2] * m, m);
                
                    d = _distance_p(
                            sdata + sindices[i] * m,
                            odata + oindices[j] * m,
                            p, m, tub);
                        
                    if (d <= tub) {
                        if (NPY_LIKELY(p == 2.0))
                            d = std::sqrt(d);
                        else if ((p != 1) && (p != infinity))
                            d = std::pow(d, 1. / p);
                         
                        coo_entry e = {sindices[i], oindices[j], d};
                        results->push_back(e);
                        
                        if (node1 == node2) {
                            coo_entry e = {sindices[j], oindices[i], d};
                            results->push_back(e);
                        }
                            
                    }
                }
            }
        }
        else {  /* 1 is a leaf node, 2 is inner node */
            tracker->push_less_of(2, node2);
            traverse(self, other, results, node1, node2->less, tracker);
            tracker->pop();
                
            tracker->push_greater_of(2, node2);
            traverse(self, other, results, node1, node2->greater, tracker);
            tracker->pop();
        }
    }        
    else {  /* 1 is an inner node */
        if (node2->split_dim == -1) {  
            /* 1 is an inner node, 2 is a leaf node*/
            tracker->push_less_of(1, node1);
            traverse(self, other, results, node1->less, node2, tracker);
            tracker->pop();
            
            tracker->push_greater_of(1, node1);
            traverse(self, other, results, node1->greater, node2, tracker);
            tracker->pop();
        }    
        else { /* 1 and 2 are inner nodes */
            tracker->push_less_of(1, node1);
            tracker->push_less_of(2, node2);
            traverse(self, other, results, node1->less, node2->less, tracker);
            tracker->pop();
                
            tracker->push_greater_of(2, node2);
            traverse(self, other, results, node1->less, node2->greater, tracker);
            tracker->pop();
            tracker->pop();
                
            tracker->push_greater_of(1, node1);
            if (node1 != node2) {
                /*
                 * Avoid traversing (node1->less, node2->greater) and
                 * (node1->greater, node2->less) (it's the same node pair
                 * twice over, which is the source of the complication in
                 * the original KDTree.sparse_distance_matrix)
                 */
                tracker->push_less_of(2, node2);
                traverse(self, other, results, node1->greater, node2->less, 
                   tracker);
                tracker->pop();
            }    
            tracker->push_greater_of(2, node2);
            traverse(self, other, results, node1->greater, node2->greater, 
                tracker);
            tracker->pop();
            tracker->pop();
        }    
    }
}



        
extern "C" PyObject*
sparse_distance_matrix(const ckdtree *self, const ckdtree *other,
                       const npy_float64 p,
                       const npy_float64 max_distance,
                       std::vector<coo_entry> *results)
{

    /* release the GIL */
    NPY_BEGIN_ALLOW_THREADS   
    {
        try {
        
            Rectangle r1(self->m, self->raw_mins, self->raw_maxes);
            Rectangle r2(other->m, other->raw_mins, other->raw_maxes);             
            RectRectDistanceTracker tracker(r1, r2, p, 0, max_distance);
            
            traverse(self, other, results, self->ctree, other->ctree, &tracker);
                                               
        } 
        catch(...) {
            translate_cpp_exception_with_gil();
        }
    }  
    /* reacquire the GIL */
    NPY_END_ALLOW_THREADS

    if (PyErr_Occurred()) 
        /* true if a C++ exception was translated */
        return NULL;
    else {
        /* return None if there were no errors */
        Py_RETURN_NONE;
    }
}