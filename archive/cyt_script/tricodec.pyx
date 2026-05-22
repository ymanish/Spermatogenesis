
from src.cyt_script.tricodec_def cimport BASE, TRIT_LEN

cdef inline unsigned int _char2trit(char ch) except *:
    ### The leading b in front of a string literal makes it a bytes object rather than a Unicode (str).

    ### b"0" (or b'0') is a length-1 bytes object whose only element is the ASCII code for the character “0” (which is 48).
    ### Indexing a bytes gives you an integer byte value:
    ### >>> b"0"[0]
    ### 48
    ### >>> b"6"
    ### b'6'
    ### >>> b"6"[0]
    ### 54      # ASCII code for '6'
    ### The same applies to b"1"[0] and b"2"[0],
    ### which are 49 and 50, respectively.
    ### The function _char2trit converts these byte values back to trit values
    ### (0, 1, and 2) by subtracting 48 from the byte value.


    ### ch is a C‐level char, which in C is also just an integer in the range 0...255.
    if ch == b'0'[0]: return 0
    elif ch == b'1'[0]: return 1
    elif ch == b'2'[0]: return 2
    raise ValueError(f"bad trit {chr(ch)}")

cpdef str int_to_tri14(unsigned int value):
    if value >= BASE ** TRIT_LEN:
        raise ValueError("out of range")
    cdef char buf[TRIT_LEN]
    cdef int pos = TRIT_LEN - 1
    while pos >= 0:
        buf[pos] = <char>(48 + value % BASE)
        value //= BASE
        pos -= 1
    return (<char*>buf)[:TRIT_LEN].decode("ascii")

cpdef unsigned int tri14_to_int(str s):
    if len(s) != TRIT_LEN:
        raise ValueError("need exactly 14 chars")
    cdef unsigned int val = 0
    cdef char ch
    for ch in s.encode("ascii"):
        val = val * BASE + _char2trit(ch)
    return val

