import rosen_2d
import numpy as np
import math
import matplotlib.pyplot as plt

def l2_norm(x):
    return np.sqrt(np.sum(x * x))

def grad_descent(X0, alpha, eps, method='Gradient descent'):
    X=X0
    grad_x=rosen_2d.rosen_2d_grad(X)
    k=0
    while(l2_norm(grad_x)>eps):
        print(X, rosen_2d.rosen_2d(X), l2_norm(grad_x))
        k+=1
        if(method=='Gradient descent'):
            u=grad_x
        if(method=='Newton'):
            H_1=np.linalg.inv(rosen_2d.rosen_2d_Hess(X))
            u = np.matmul(H_1,grad_x.T)
        X=X-alpha*u
        grad_x=rosen_2d.rosen_2d_grad(X)
    return k

def q5():
    step = [0.001, 0.002, 0.005]
    eps = [0.1, 0.01, 0.001, 0.0001]
    X0=np.array([1.2,1.2])
    for alpha in step:
        for e in eps:
            k = grad_descent(X0, alpha, e)
            print(f'Grad stepsize = {alpha}, epsilon = {e}, iter: {k}')

    for e in eps:
        k = grad_descent(X0, 1, e, method='Newton')
        print(f'Newton epsilon = {e}, iter: {k}')

    k = grad_descent(X0, 0.005, 0.1)
    return

def q6():
    alg1 = lambda k: 1+1/k
    alg2 = lambda k: 1+(0.5**(2**k))
    alg3 = lambda k: 1+1/math.factorial(k)
    alg4 = lambda x,k: x[k-1]-(0.5**k)

    n_iter =11
    v_alg1 = np.zeros(n_iter)
    v_alg2 = np.zeros(n_iter)
    v_alg3 = np.zeros(n_iter)
    v_alg4 = np.zeros(n_iter)
    v_alg4[0]=2
    for k in range(1,n_iter):
        v_alg1[k] = alg1(k)
        v_alg2[k] = alg2(k)
        v_alg3[k] = alg3(k)
        v_alg4[k] = alg4(v_alg4, k)

    x= range(1,n_iter)
    fig, ax = plt.subplots()
    ax.plot(x, v_alg1[1:], label = 'alg1')
    ax.plot(x, v_alg2[1:], label = 'alg2')
    ax.plot(x, v_alg3[1:], label = 'alg3')
    ax.plot(x, v_alg4[1:], label = 'alg4')
    ax.legend()


    eps =0.1
    k=1
    z=alg3(k)
    #z=2-(0.5**k)
    while( not z<=(1+eps)):
        k+=1
        z=alg3(k)
        print(k,z)
        #z=z-(0.5**k)
    print(k)
    return

def bisection(a,b,N, tol):
    f= lambda x: 4*(x**3) - 42*(x**2) + 120*x -70
    x_left=a
    x_right=b
    for i in range(1,N+1):
        x_mid = (x_left+x_right)/2
        print(i,x_mid, x_left, x_right, (x_right - x_left)/2)
        if(f(x_mid)==0 or (x_right - x_left)/2<=tol):
            print('Ending')
            print('mid, f(mid)  , tol')
            print(x_mid , f(x_mid), (x_right - x_left)/2)
            return [x_mid, i]
        if(f(x_mid)*f(x_left)<0):
            x_right=x_mid
        if(f(x_mid)*f(x_left)>0):
            x_left=x_mid
    return [x_mid,i]


def q7():

    x = np.arange(0,7,0.001)
    f = x**4 -14*(x**3) + 60*(x**2) - 70*x
    f1 = 4*(x**3) - 42*(x**2) + 120*x -70
    plt.plot(x, f,label = 'f')
    plt.plot(x, f1, label='der')
    plt.legend()
    plt.show()

    x_mid, iter = bisection(0,2,20,0.001)
    print(x_mid, iter)

def quad_grad_descent():
    eps=0.0001
    alpha=0.2
    X0 = np.array([-135.1150, -4.5224, 130.1168, -5.6879])
    b = np.array([0.4218, 0.9157, 0.7922, 0.9595])
    Q = np.array([[2.3346, 1.1384, 2.5606, 1.4507], [1.1384, 0.7860, 1.2743, 0.9531],\
                [2.5606, 1.2743, 2.8147, 1.6487], [1.4507, 0.9531, 1.6487, 1.8123]])
    x_opt = np.matmul(np.linalg.inv(Q), b)
    evals, evecs = np.linalg.eig(Q)
    print(evals)
    print(evecs)
    print(x_opt-X0)
    print(f'alpha_onestep: {1/evals[0]}')

    x=X0
    k=0
    while(l2_norm(x-x_opt)>eps):
        k+=1
        x=x-alpha*(np.matmul(Q,x)-b)
        print(k,l2_norm(x-x_opt))
    print(f'\n\nResults for alpha={alpha}, epsilon={eps} :')

    print(f'diff_x: {l2_norm(x-x_opt)}')
    print(f'Iterations: {k}')
    print(f'Min at {x}')
    return k

def main():
    #q5()
    #q6()
    #q7()
    quad_grad_descent()
    #k = grad_descent(np.array([1.2,1.2]), 0.005, 0.0001)
    #print(k)


    return

if __name__ == '__main__':
    main()

