import math


def sin(x):
    s0 = 0
    sn = x
    e = 1e-5  # 误差
    n = 2
    while abs((sn-s0)/sn)>e:
        s0 = sn
        sn = (-1)**(n-1)*(x**(2*n-1)/math.factorial(2*n-1)) + sn
        n += 1
    return sn, n


def integral(n):
    i0 = 0
    iN = i0
    for i in range(n-1, 0, -1):
        i0 = iN
        iN = i0/(-5) + 1/(5*i)
    return iN
