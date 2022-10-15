from lib.Preprocessing import create_word_database, extract_features
from lib.Partition_And_Summarise import summarise_labels
from lib.Testing import accuracy, get_classification


# ---------------------------------------------------------------------------------
# --- DRIVER PROGRAM ---
# ---------------------------------------------------------------------------------

def naive_bayes_classifier(train_dir ="../train-mails", test_dir ="../test-mails"):
    top_words, top_word_id = create_word_database(train_dir)
    train_features_matrix, train_email_labels = extract_features(train_dir, top_word_id)
    test_feature_matrix, test_labels = extract_features(test_dir, top_word_id)

    model = summarise_labels(train_features_matrix, train_email_labels)
    predicted_labels = get_classification(model, test_feature_matrix)
    print("Classifier: Finished classification.")
    print("Classifier: Accuracy score = " + str(accuracy(test_labels, predicted_labels)))


# naive_bayes_classifier()
