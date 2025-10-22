from ast import literal_eval


def simplest_type(s: str):
    try:
        return literal_eval(s)
    except:
        return s
