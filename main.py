from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

from knn import knn, knn_sklearn
from knn_error import knn_error


# Load dataset
iris = load_iris()
X = iris.data
Y = iris.target

# KNN prediction
Ypred = knn(X, Y)

# Error percentage
knn_error(Ypred, Y)

# Use sklearn library
average_errors = []
knn_sklearn(X, Y, 1, average_errors)
average_error = average_errors[0]

print("Error while using KNeighborsClassifier function:")
print("k=1:", f'{average_error:.2f}' + "% of error")

# Iterate through different value of k
for k in range(2, 51):
    knn_sklearn(X, Y, k, average_errors)

# Plot the results
plt.plot(range(1, 51), average_errors, color='blue', linestyle='dashed',
    marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Percentage for K in range(1, 51)')
plt.xlabel('K')
plt.ylabel('Error Percentage')
plt.show()
