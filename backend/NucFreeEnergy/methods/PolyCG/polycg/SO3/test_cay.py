import numpy as np

from so3 import cayley2euler, cayley2euler_factor, euler2cayley, euler2cayley_factor

omega = np.array([3, -4, -3])
phi = np.array([8, -2, 36])
conv = np.pi / 180

omega = omega * conv
phi = phi * conv

euler = phi + omega
# euler = euler * conv

cay = euler2cayley(euler)
cays = euler2cayley(phi)
cayd = cay - cays

cay_phi0 = euler2cayley_factor(phi) * euler
cays_phi0 = euler2cayley_factor(phi) * phi
cayd_phi0 = euler2cayley_factor(phi) * omega


print(cay / conv)
print(cay_phi0 / conv)


print(cays / conv)
print(cays_phi0 / conv)

print(cayd / conv)
print(cayd_phi0 / conv)
