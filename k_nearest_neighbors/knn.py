from statistics import mode

from numpy import unique, delete, sum, argsort, shape
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def knn(x, y, k):
    y_predicted = []
    # Find the nearest neighbor of all dataset and sort the first k elements
    nearest_neighbor = pairwise_distances(x, x)
    arg = argsort(nearest_neighbor)[:, 1:(k+1)] # start at 1 because 0 is the data itself

    # Affect the label
    predictions = y[arg]

    for i in range(len(predictions)):
        # Find the mode
        prediction = mode(predictions[i])
        y_predicted.append(prediction)
    return y_predicted

def knn_sklearn(x, y, k, average_errors):
    # Creation of the model with different k
    sklearn_knn_model = KNeighborsClassifier(n_neighbors=k)
    # Compute the accuracy with the cross validation of type leave one out
    accuracy = cross_val_score(sklearn_knn_model, x, y, cv=LeaveOneOut())
    # Convert the accuracy to a error percentage
    average_error = (1 - accuracy.mean()) * 100
    # Add the error percentage to the list
    average_errors.append(average_error)
    return

