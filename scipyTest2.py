# 2--optimizer
import numpy as np
from scipy.optimize import minimize


def rosen(x):
    return sum(100.0 * (x[1:] - x[:1] ** 2.0) ** 2.0 + (1 - x[:1]) ** 2.0)


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method="nelder-mead", options={"xtol": 1e-8, "disp": True})
print("ROSE MINI:", res.x)


def func(x):
    return -(2 * x[0] * x[1] + 2 * x[0] - x[0] ** 2-2 * x[1] ** 2)


def func_deriv(x):
    dfdx0 = -(2 * x[1] + 2 - 2 * x[0])
    dfdx1 = -(2 * x[0] - 4 * x[1])
    return np.array([dfdx0, dfdx1])
cons = ({"type": "eq", "fun": lambda x: np.array([x[0]**3-x[1]]), "jac": lambda x:np.array([3.0*(x[0]**2), -1.0])},
        dict(type="ineq", fun=lambda x: np.array([x[1] - 1]), jac=lambda x: np.array([0.0, 1.0])),)
res = minimize(func, [-1.0, 1.0], jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
print("RESTRICT:", res)

# root
from scipy.optimize import root


def fun(x):
    return x + 2 * np.cos(x)
sol = root(fun, 0.1)
print("Root:", sol.x, sol.fun)
