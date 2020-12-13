
import numpy as np

def rosen_2d(X):
    x = X[0]
    y = X[1]
    return 5 * (x*x + y*y) - x*y -11*x +11*y +11

def rosen_2d_grad(X):
    x = X[0]
    y = X[1]
    dx = 10*x - y - 11
    dy = 10*y - x + 11
    return np.array([dx,dy])

def rosen_2d_Hess(X):
    return np.array([[10, -1], [-1, 10]])

def main():
    if __name__ == '__main__':
        main()