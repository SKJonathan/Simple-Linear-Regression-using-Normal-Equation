import numpy as np
import matplotlib.pyplot as plt

#Simple Linear Regressionusing using Normal Equation
#Data

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 4.5],
    [1, 4.3],
    [1, 6],
])

y = np.array([5,7,9,11,13,10,9,16])


# Compute the best-fitting weights using the Normal Equation
def FindingWeights(X, y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    #print("Computed weights (w):", w)

    return w





w = FindingWeights(X, y)



#Creates 100 points between the smallest and the largest x point.
x_line = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
y_line = w[0] + w[1] * x_line  # y = bias + slope * x

plt.figure(figsize=(8,5))  # Bigger plot for clarity
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability


# Creates the line
plt.plot(x_line, y_line, color='red', label='Regression line')
# Creates Points
# in X[:,1] : Says go through all points
plt.scatter(X[:,1], y, color='blue', label='Data points')
plt.xlabel('Feature X1')
plt.ylabel('Target Y')
plt.title('Simple Linear Regression Fit')
plt.legend()
plt.show()



