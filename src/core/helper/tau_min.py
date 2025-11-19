
def _compute_tau_min(k_wrap: float, ends: int = 2, gamma: float = 5.0) -> float:
    # If either end can initiate rewrap from fully unwrapped, ends=2
    import math
    w0 = ends * float(k_wrap)
    t099 = math.log(100.0) / w0
    return gamma * t099