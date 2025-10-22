#!/bin/env python3

"""
    Methods that allow numba decorators that will revert back to native python if the numba module fails to load
"""

def conditional_numba(function):
    """
        The conditional decorator will use numba if the package is installed 
        and otherwise revert to native python
    """
    try:
        from numba import jit
        return jit(nopython=True)(function)
    except ModuleNotFoundError:
        print("Warning: {function.__name__}: numba not installed. For speedup please install numpy: pip install numba")
        return function
    
def conditional_jitclass(origclass):
    """
        The conditional decorator will use numba if the package is installed 
        and otherwise revert to native python
    """
    try:
        from numba.experimental import jitclass
        return jitclass()(origclass)
    except ModuleNotFoundError:
        print("Warning: {function.__name__}: numba not installed. For speedup please install numpy: pip install numba")
        return origclass