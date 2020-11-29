
import numpy as np

def rosen_2d(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.

def rosen_2d_grad(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return np.array([-2*a - 400*b*x, 200*b])

def rosen_2d_Hess(X):
    x = X[0]
    y = X[1]
    h11=2-400*y + 1200*x*x
    h12=-400*x
    h22=200
    return np.array([[h11,h12], [h12,h22]])

def main():
    if __name__ == '__main__':
        main()