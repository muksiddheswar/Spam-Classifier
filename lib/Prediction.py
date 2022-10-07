# ---------------------------------------------------------------------------------
# --- PREDICTION ---
# ---------------------------------------------------------------------------------

import math
import json

# Calculates the probability as per normal distribution
# def calculate_gaussian_probability(x, mean, stdev):
#     exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
#     return (1 / (math.sqrt(2*math.pi) * stdev)) * exp


def calculate_log_gaussian_probability(x, mean, stdev):
    # Calculates the log of the terms as per normal distribution
    exponent = math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))
    ft = math.log(1 / (math.sqrt(2 * math.pi) * stdev))
    return ft, exponent


# Calculates probability of an instance being a part of each class.
def check_class_probabilities(features_summary, input_instance):
    probabilities = {}
    for class_label, class_summary in features_summary.items():

        log_terms = 0
        exponent_terms = 0

        for i, summary in list(enumerate(class_summary)):
            mean, stddev = summary
            x = input_instance[i]
            # If it turns out that some feature has 0 variance, we are going to ignore that.
            if stddev != 0:
                ft, exponent = calculate_log_gaussian_probability(x, mean, stddev)
                log_terms += ft
                exponent_terms += exponent

        f = open("mail_count.txt", "r")
        mail_count = int(f.read())
        f.close()

        f = open("class_count.txt", "r")
        result = f.read()
        class_count = json.loads(result)
        f.close()

        probabilities[str(class_label)] = log_terms - exponent_terms \
                                          + math.log((class_count[str(class_label)] / mail_count))
    return probabilities


# Predicts the class of the function
def classify(features_summary, input_instance):
    probabilities = check_class_probabilities(features_summary, input_instance)

    final_label, max_prob = -1, 1
    for label, prob in probabilities.items():
        # Boundary condition
        if final_label == -1:
            final_label, max_prob = float(label), prob

        if prob > max_prob:
            final_label, max_prob = float(label), prob
    return final_label
