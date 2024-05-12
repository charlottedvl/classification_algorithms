import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def knn(x, y):
    y_predicted = []
    # For each element in x, we delete the element and the label corresponding in y
    # in order to achieve cross validation of leave one type
    for i in range(len(x)):
        training_x = np.delete(x, i, axis=0)
        training_y = np.delete(y, i)

        # Reshape as a row so we have the same dimension
        x_i = x[i].reshape(1, -1)

        # Find the nearest neighbor of the x[i] element inside the training set
        nearest_neighbor = pairwise_distances_argmin(x_i, training_x)
        # Affect the label
        prediction = training_y[nearest_neighbor[0]]
        # Add the prediction to the list y_predicted
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

