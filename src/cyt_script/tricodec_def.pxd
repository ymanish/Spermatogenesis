
cdef enum:
    BASE     = 3
    TRIT_LEN = 14

# if you want to cimport your C‐helpers, add these too:
cdef inline unsigned int _char2trit(char ch) except *

# and your Python‐visible functions (so cimport can see the signatures):
cpdef str int_to_tri14(unsigned int value)
cpdef unsigned int tri14_to_int(str s)

cpdef unsigned int edit_tricodec(unsigned int value, int position, int new_digit) except *