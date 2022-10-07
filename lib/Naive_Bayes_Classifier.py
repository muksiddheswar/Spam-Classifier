from lib.Preprocessing import make_dictionary, extract_features
from lib.TrainingAndSummarise import summarise_class
from lib.Testing import accuracy, get_classification

# TRAIN_DIR = "../train-mails"
# TEST_DIR = "../test-mails"


# ---------------------------------------------------------------------------------
# --- DRIVER PROGRAM ---
# ---------------------------------------------------------------------------------


def naive_bayes_classifier(train_dir ="../train-mails", test_dir ="../test-mails"):
    dictionary = make_dictionary(train_dir)
    print("reading and processing emails from file.")

    features_matrix, labels = extract_features(train_dir)
    test_feature_matrix, test_labels = extract_features(test_dir)

    print("Training model.")
    model = summarise_class(features_matrix, labels)

    predicted_labels = get_classification(model, test_feature_matrix)

    print("FINISHED classifying. accuracy score : ")
    print(accuracy(test_labels, predicted_labels))


# dictionary = make_dictionary(TRAIN_DIR)
# print("reading and processing emails from file.")
#
# features_matrix, labels = extract_features(TRAIN_DIR)
# test_feature_matrix, test_labels = extract_features(TEST_DIR)
#
# print("Training model.")
# model = summarise_class(features_matrix, labels)
#
# predicted_labels = get_classification(model, test_feature_matrix)
#
# print("FINISHED classifying. accuracy score : ")
# print(accuracy(test_labels, predicted_labels))
