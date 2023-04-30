import numpy as np


def objective(params, data, lamb):
    x, y, t, s = params
    xi, yi, ti = data
    n = len(xi)

    g = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                d1 = (x - xi[i]) ** 2 + (y - yi[i]) ** 2
                d2 = (x - xi[j]) ** 2 + (y - yi[j]) ** 2
                term = ((t - (ti[j] - s)) ** 2 * d1 - (t - (ti[i] - s)) ** 2 * d2)
                g += term ** 2
    return g + lamb * np.abs(s)


def gradient(params, data, lamb):
    x, y, t, s = params
    xi, yi, ti = data
    n = len(xi)

    grad = np.zeros(4)
    for i in range(n):
        for j in range(n):
            if i != j:
                d1 = (x - xi[i]) ** 2 + (y - yi[i]) ** 2
                d2 = (x - xi[j]) ** 2 + (y - yi[j]) ** 2

                term = ((t - (ti[j] - s)) ** 2 * d1 - (t - (ti[i] - s)) ** 2 * d2)

                grad[0] += 4 * term * (2 * (x - xi[j]) * (t - ti[j] + s) * d1 - 2 * (x - xi[i]) * (t - ti[i] + s) * d2)
                grad[1] += 4 * term * (2 * (y - yi[j]) * (t - ti[j] + s) * d1 - 2 * (y - yi[i]) * (t - ti[i] + s) * d2)
                grad[2] += 4 * term * ((t - ti[j] + s) * d1 - (t - ti[i] + s) * d2)
                grad[3] += 4 * term * (2 * (t - ti[j] + s) * d1 - 2 * (t - ti[i] + s) * d2) - 2 * lamb

    return grad

def gradient_descent(data, lamb, initial_lr=0.01, c=1e-4, lr_reduction_factor=0.5, max_iter=1000, tol=1e-6):
    x, y, t = data
    params = np.zeros(4)
    objective_prev = np.inf
    lr = initial_lr

    for _ in range(max_iter):
        grad = gradient(params, data, lamb)
        new_params = params - lr * grad
        new_objective = objective(new_params, data, lamb)

        while new_objective > objective_prev - c * lr * np.linalg.norm(grad) ** 2:
            lr *= lr_reduction_factor
            new_params = params - lr * grad
            new_objective = objective(new_params, data, lamb)

        if new_objective < objective_prev - tol:
            params = new_params
            objective_prev = new_objective
        else:
            break

    return params


data1 = (
    np.array([-3, -1, 1, 3, -3, -1, 1, 3]),
    np.array([1, 1, 1, 1, -1, -1, -1, -1]),
    np.array([2.01556443707464, 1.03077640640442, 0.25, 1.03077640640442, 2.13600093632938, 1.25, 0.75, 1.25])
)

data2 = (
    np.array([-3, -1, 1, 3, -3, -1, 1, 3]),
    np.array([1, 1, 1, 1, -1, -1, -1, -1]),
    np.array(
        [2.01859964502113, 1.02477314078308, 0.254899653211739, 1.03817003764046, 2.1531198141592, 1.24805876464242,
         0.728616447305601, 1.24160411252663])
)

lamb_values = [0.1, 0.01, 0.001]

for lamb in lamb_values:
    result1 = gradient_descent(data1, lamb)
    result2 = gradient_descent(data2, lamb)

    print(f"Results for lamb={lamb}:")
    print("Result for data1(x,y,t,s):", result1)
    print("Result for data2(x,y,t,s):", result2)
    print()

lamb_values = np.linspace(0, 10, 11)

for lamb in lamb_values:
    result1 = gradient_descent(data1, lamb)
    result2 = gradient_descent(data2, lamb)
    print(f"Results for lamb = {lamb}:")
    print("Result for data1(x,y,t,s):", result1)
    print("Result for data2(x,y,t,s):", result2)
    print("\n")