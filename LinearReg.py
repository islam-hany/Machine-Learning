import numpy as np
import matplotlib.pyplot as plt

def Cost_Function(w, b, x: np.ndarray, y: np.ndarray):
    return (1 / (2.0 * x.shape[0])) * np.sum((w * x + b - y) ** 2)

def gradient_decent(x: np.ndarray, y: np.ndarray, alpha: float, error: float, max_iter=10000):
    w = b = 0
    Cost_func = 10000
    costs = []
    iterations = 0

    while Cost_func > error and iterations < max_iter:
        sum_w = 0
        sum_b = 0
        for j in range(x.shape[0]):
            sum_w += (1 / float(x.shape[0])) * (w * x[j] + b - y[j]) * x[j]
            sum_b += (1 / float(x.shape[0])) * (w * x[j] + b - y[j])

        w -= alpha * sum_w
        b -= alpha * sum_b

        Cost_func = Cost_Function(w, b, x, y)
        costs.append(Cost_func)
        iterations += 1

    return w, b, costs , iterations

# Dataset
x = np.array(range(1,11))
y = np.random.randint(0,50,x.shape[0])

plt.scatter(x, y, marker='x', c='r')

w, b, costs , iterations = gradient_decent(x, y, 0.01, 0.25)

# Plot fitted line
model = w * x + b
plt.plot(x, model, label=f"y = {w:.2f}x + {b:.2f}")
plt.legend()
plt.show()

# Plot cost convergence
plt.plot(range(1,iterations+1),costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")


plt.show()
