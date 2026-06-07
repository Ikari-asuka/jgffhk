import numpy as np

def f1(x):
    return 3*x[0]**2 + 2*x[1]**2 - 4*x[0] - 6*x[1]

def grad_f1(x):
    return np.array([6*x[0] - 4, 4*x[1] - 6])

def steepest_descent(f, grad_f, x0, eps=1e-5, max_iter=100):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < eps:
            break
        # 固定步长(或用精确线搜索，这里用最优步长)
        alpha = (g @ g) / (g @ np.array([[6,0],[0,4]]) @ g)
        x = x - alpha * g
    return x, f(x)

# 求解
x0 = [0, 0]
x_opt, f_opt = steepest_descent(f1, grad_f1, x0)
print(f"最优解：x = {x_opt}, f(x) = {f_opt}")
