from numpy import unique, where, linalg, mean


def nbc(x, y):
    probabilities = {}
    class_probabilities = {}
    barycentres = []
    # Calculate class probabilities and store class lists
    for one_class in unique(y):
        indices = where(y == one_class)[0]
        one_class_list = y[indices]
        extracted_data = x[indices]
        class_probability = len(one_class_list) / len(y)
        barycenter = mean(extracted_data, axis=0)
        barycentres.append(barycenter)
        class_probabilities[one_class] = [class_probability, one_class_list, barycenter]
    # Predict class for each data point in x
    y_predicted = []
    for data in x:
        print("                  ")
        predicted_class = None
        predicted_probability = 0
        for one_class in class_probabilities:
            class_probability, one_class_list, barycenter = class_probabilities[one_class]
            distance = linalg.norm(data - barycenter)
            distance_sum = sum([linalg.norm(data - barycenter) for barycenter in barycentres])
            print(distance)
            print(distance_sum)
            print(class_probability)
            conditional_probability = distance / distance_sum
            print(conditional_probability)
            probability = conditional_probability * class_probability
            print(probability)
            if predicted_class is None or probability > predicted_probability:
                predicted_probability = probability
                predicted_class = one_class
        y_predicted.append(predicted_class)
    print(y_predicted)
    return y_predicted
