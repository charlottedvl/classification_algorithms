def knn_error(y_pred, y):
    error = (1 - sum(y == y_pred)/len(y)) * 100
    print(f'{error:.2f}' + "% of error")
    return error
