import numpy as np
from numpy import exp, pi, sin, cos, sqrt, e    


def ackley(x):
    """
    Ackley function
    - arbitrary n_var
    - exploration range [-10,10] for each variable
    - global minimum at x_0 = (0,...,0), ackley(x_0) = 0
    """
    n_var = x.shape[-1]
    a = -20*exp(-0.2 * sqrt(np.sum(x**2, axis = -1)/n_var))
    b = -exp(np.sum(cos(2*pi*x), axis = -1)/n_var)
    return a + b + 20 + e


def alpine1(x):
    """
    Alpine function #1
    - arbitrary n_var
    - exploration range [-10,10] for each variable
    - global minimum at x_0 = (0,...,0), alpine(x_0) = 0
    """
    return np.sum(
        np.abs(x*sin(x) + 0.1*x),
        axis = -1
    )


def alpine2(x):
    """
    Alpine function #2
    - arbitrary n_var
    - exploration range [0,10] for each variable
    - global maximum at x_0 = 7.9170526982459462172*(1,...,1), alpine2(x_0) = 2.80813118000070053291 ** n_var
    """
    return np.sum(
        sqrt(x)*sin(x),
        axis = -1
    )


def booth(x):
    """
    Booth function
    - n_var = 2
    - exploration range [-10,10] for each variable
    - global minimum at x_0 = (1,3), booth(x_0) = 0
    """
    n_var = 2
    if x.shape[-1] != n_var:
        raise ValueError("Inputs must be 2D vectors or arrays thereof.")
    x1, x2 = x.T
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2


def colville(x):
    """
    Colville function
    - n_var = 4
    - exploration range [-10,10] for each variable
    - global minimum at x_0 = (1,...,1), colville(x_0) = 0
    """
    n_var = 4
    if x.shape[-1] != n_var:
        raise ValueError("Inputs must be 4D vectors or arrays thereof.")
    x1, x2, x3, x4 = x.T
    a = 100*(x2 - x1**2)**2 + (1 - x1)**2
    b = 90*(x4 - x3**2)**2 + (1 - x3)**2
    c = 10.1*((x2 - 1)**2 + (x4 - 1)**2)
    d = 19.8*(x2 - 1)*(x4 - 1)
    return a + b + c + d


def goldstein_price(x):
    """
    Goldstein-Price function
    - n_var = 2
    - exploration range [-2,2] for each variable
    - global minimum at x_0 = (0,-1), goldstein_price(x_0) = 3
    """
    n_var = 2
    if x.shape[-1] != n_var:
        raise ValueError("Inputs must be 2D vectors or arrays thereof.")
    x1, x2 = x.T
    a = (x1 + x2 + 1)**2
    b = (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    c = (2*x1 - 3*x2)**2
    d = (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return (1 + a*b)*(30 + c*d)


def matyas(x):
    """
    Matyas function
    - n_var = 2
    - exploration range [-10,10] for each variable
    - global minimum at x_0 = (0,0), matyas(x_0) = 0
    """
    n_var = 2
    if x.shape[-1] != n_var:
        raise ValueError("Inputs must be 2D vectors or arrays thereof.")
    x1, x2 = x.T
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2


def rastrigin(x):
    """
    Rastrigin function
    - arbitrary n_var
    - exploration range [-5.12,5.12] for each variable
    - global minimum at x_0 = (0,...,0), rastrigin(x_0) = 0
    """
    n_var = x.shape[-1]
    r = np.sum(
        x**2 - 10*cos(2*pi*x), 
        axis = -1
    )
    return 10*n_var + r


def rosenbrock(x):
    """
    Rosenbrock function
    - arbitrary n_var
    - exploration range [-2,2] for each variable
    - global minimum at x_0 = (1,...,1), rosenbrock(x_0) = 0
    """
    n_var = x.shape[-1]
    r = 0
    for i in range(n_var-1):
        r += 100*(x[:,i+1] - x[:,i]**2)**2 + (x[:,i] - 1)**2
    return r 


def schwefel(x):
    """
    Schwefel function
    - arbitrary n_var
    - exploration range [-500,500] for each variable
    - global minimum at x_0 = 420.9687*(1,...,1), schwefel(x_0) = 0
    """
    n_var = x.shape[-1]
    s = np.sum(
        x*sin(sqrt(np.abs(x))), 
        axis = -1
    )
    return 418.9828872724338*n_var - s