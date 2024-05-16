from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

from k_nearest_neighbors.knn import knn, knn_sklearn
from k_nearest_neighbors.knn_error import knn_error
from naive_bayesian_classifier.nbc import nbc_barycenters, nbc_sklearn, nbc_gaussian_distribution

# Load dataset
iris = load_iris()
X = iris.data
Y = iris.target

# KNN prediction
Ypred = knn(X, Y, 20)

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

y_pred_barycenter = nbc_barycenters(X, Y)
knn_error(y_pred_barycenter, Y)

y_pred_sklearn = nbc_sklearn(X, Y)
knn_error(y_pred_sklearn, Y)

y_pred_gauss = nbc_gaussian_distribution(X, Y)
knn_error(y_pred_gauss, Y)

