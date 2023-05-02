import numpy as np

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

x1, y1, t1 = data1
x2, y2, t2 = data2

def objective(x, y, t, s, X, Y, T, S):

    n = len(x)

    g = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                term = ((T - (t[j] - S[j])) ** 2 * ((X - x[i]) ** 2 + (Y - y[i]) ** 2) - (T - (t[i] - S[i])) ** 2 * ((X - x[j]) ** 2 + (Y - y[j]) ** 2)) ** 2
                g += term
    return g

def decay_optimization(initial_lr, decay, iteration):
    return initial_lr * (decay ** iteration)


def gradient(x,y,t,lamb, learning_rate, decay, num):
    np.random.seed(100)
    X, Y, T = np.random.rand(3)
    S = np.random.rand(len(x))
    for _ in range(num):
        dX, dY, dT = np.zeros(3)
        dS = np.zeros(len(x))
        n = len(x)
        clip_value = 0.1
        for i in range(n):
            for j in range(n):
                if i != j:
                    term = ((T - (t[j] - S[j])) ** 2 * ((X - x[i]) ** 2 + (Y - y[i]) ** 2) - (
                                T - (t[i] - S[i])) ** 2 * ((X - x[j]) ** 2 + (Y - y[j]) ** 2)) ** 2
                    dX += term * (2 * (X - x[i]) * (T - t[i] + S[i]) - 2 * (X - x[j]) * (T - t[j] + S[j]))
                    dY += term * (2 * (Y - y[i]) * (T - t[i] + S[i]) - 2 * (Y - y[j]) * (T - t[j] + S[j]))
                    dT += term * ((T - t[j] + S[j]) * ((X - x[i]) ** 2 + (Y - y[i]) ** 2) - (T - t[i] + S[i]) * (
                                (X - x[j]) ** 2 + (Y - y[j]) ** 2))
                    dS[i] += term * (2 * (T - t[j] + S[j]) * ((X - x[i]) ** 2 + (Y - y[i]) ** 2) - 2 * (T - t[i] + S[i]) * (
                                (X - x[j]) ** 2 + (Y - y[j]) ** 2))

                dX = np.clip(dX, -clip_value, clip_value)
                dY = np.clip(dY, -clip_value, clip_value)
                dT = np.clip(dT, -clip_value, clip_value)
                dS = np.clip(dS, -clip_value, clip_value)

                learning_rate = decay_optimization(learning_rate, decay, _)

                X -= learning_rate * dX
                Y -= learning_rate * dY
                T -= learning_rate * dT
                S -= learning_rate * dS + 2 * lamb * S

        return X, Y, T, S

def optimize(x, y, t, lamb, learning_rate, decay, num):
    X, Y, T, S = gradient(x, y, t, lamb, learning_rate, decay, num)
    return X, Y, T, S

lambdas = np.arange(0, 5, 0.1)
best_lambda_value = 0
min_objective_value = float('inf')

for lamb in lambdas:
    X1, Y1, T1, S1 = optimize(x1, y1, t1, lamb, learning_rate=0.01, decay=0.01, num=1000)
    X2, Y2, T2, S2 = optimize(x2, y2, t2, lamb, learning_rate=0.01, decay=0.01, num=1000)

    obj_data1 = objective(x1, y1, t1, S1, X1, Y1, T1, S1)
    obj_data2 = objective(x2, y2, t2, S2, X2, Y2, T2, S2)
    current_objective_value = obj_data1 + obj_data2 + lamb * (np.sum(np.abs(S1)) + np.sum(np.abs(S2)))

    if current_objective_value < min_objective_value:
        min_objective_value = current_objective_value
        best_lambda_value = lamb
        best_X1, best_Y1, best_T1, best_S1 = X1, Y1, T1, S1
        best_X2, best_Y2, best_T2, best_S2 = X2, Y2, T2, S2

print(f"Best lambda value: {best_lambda_value}")
print(f"Result for data1(x,y,t,s): X = {best_X1}, Y = {best_Y1}, T = {best_T1}, S = {best_S1}")
print(f"Result for data1(x,y,t,s): X = {best_X2}, Y = {best_Y2}, T = {best_T2}, S = {best_S2}")



