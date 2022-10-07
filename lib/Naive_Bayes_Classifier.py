from Preprocessing import make_dictionary, extract_features
from TrainingAndSummarise import summarise_class
from Testing import accuracy, get_classification

TRAIN_DIR = "../train-mails"
TEST_DIR = "../test-mails"

word_id = {}
mail_count = 0
class_count = {}


# ---------------------------------------------------------------------------------
# --- DRIVER PROGRAM ---
# ---------------------------------------------------------------------------------


def naive_bayes_classifier(TRAIN_DIR, TEST_DIR):
    dictionary = make_dictionary(TRAIN_DIR)
    print("reading and processing emails from file.")

    features_matrix, labels = extract_features(TRAIN_DIR)
    test_feature_matrix, test_labels = extract_features(TEST_DIR)

    print("Training model.")
    model = summarise_class(features_matrix, labels)

    predicted_labels = get_classification(model, test_feature_matrix)

    print("FINISHED classifying. accuracy score : ")
    print(accuracy(test_labels, predicted_labels))


dictionary = make_dictionary(TRAIN_DIR)
print("reading and processing emails from file.")

features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

print("Training model.")
model = summarise_class(features_matrix, labels)

predicted_labels = get_classification(model, test_feature_matrix)

print("FINISHED classifying. accuracy score : ")
print(accuracy(test_labels, predicted_labels))
