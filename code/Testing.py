from Prediction import classify

# ---------------------------------------------------------------------------------
# --- TESTING ---
# ---------------------------------------------------------------------------------


def get_classification(features_summary , input_instances):
    classifications = []
    for i, input in list(enumerate(input_instances)):
        prediction = classify(features_summary, input)
        classifications.append(prediction)
    return classifications


def accuracy(test_labels , predicted_labels):
    true_positive = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predicted_labels[i]:
            true_positive += 1
    score = true_positive / float(len(test_labels))
    return (score*100)