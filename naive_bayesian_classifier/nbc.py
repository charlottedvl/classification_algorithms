from numpy import unique, where, linalg, mean


def nbc(x, y):
    # Store the probability of each class and the associated sublist
    class_probabilities = {}
    # Barycenters of each class
    barycenters = {}

    for one_class in unique(y):
        # Extract the sublists corresponding to the class
        indices = where(y == one_class)[0]
        one_class_list = y[indices]
        extracted_data = x[indices]
        # P(class) = (Number of element in the class) / (Number total of element)
        class_probability = len(one_class_list) / len(y)
        class_probabilities[one_class] = [class_probability, one_class_list]
        # Calculate barycenter of the class
        barycenter = mean(extracted_data, axis=0)
        barycenters[one_class] = barycenter

    # List of predicted class
    y_predicted = []

    for data in x:
        predicted_class = None
        predicted_probability = 0
        # Calculate the sum of the distances between the data and the barycenter of each class
        distance_sum = sum([linalg.norm(data - barycenters[one_class]) for one_class in barycenters])

        for one_class in class_probabilities:
            class_probability, one_class_list = class_probabilities[one_class]
            # Calculate the distance between the data and the barycenter of the class
            distance = linalg.norm(data - barycenters[one_class])
            # P(data | class) = 1 - ((distance(data -> barycenter)) / (sum(distances(data -> each barycenter))))
            conditional_probability = 1 - (distance / distance_sum)
            # P(class | data) = P(data | class) * P(class)
            probability = conditional_probability * class_probability
            # Update the prediction if necessary
            if predicted_class is None or probability > predicted_probability:
                predicted_probability = probability
                predicted_class = one_class
        # Add the prediction
        y_predicted.append(predicted_class)
    return y_predicted
