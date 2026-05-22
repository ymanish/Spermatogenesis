from src.cyt_script.tricodec import int_to_tri14, tri14_to_int
## 
from src.cyt_script.tricodec_def cimport TRIT_LEN, BASE

cpdef unsigned int edit_tricodec(unsigned int value, int position, int new_digit) except *:
    """Convert an integer to a tricodec string, change the position to new_digit (0,1,2) in tricodec and then return back to an integer.
    tri_value = int_to_tri14(value)
    tri_value = tri_value[:position] + str(new_digit) + tri_value[position+1:]
    return tri14_to_int(tri_value) """

    cdef str tri_value
    cdef str new_tri

    # bounds checks
    if position < 0 or position >= TRIT_LEN:
        raise IndexError(f"position {position} out of range 0..{TRIT_LEN-1}")
    if new_digit < 0 or new_digit >= BASE:
        raise ValueError(f"new_digit must be 0..{BASE-1}")

    # 1) encode to a 14‐char string
    tri_value = int_to_tri14(value)
    # 2) splice in the new trit character
    new_tri = (
        tri_value[:position]
        + chr(48 + new_digit)          # ASCII '0'+new_digit
        + tri_value[position+1:]
    )
    # 3) decode back to int
    return tri14_to_int(new_tri)
