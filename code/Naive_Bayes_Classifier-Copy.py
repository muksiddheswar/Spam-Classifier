import os
import math
import numpy as np
from collections import Counter
TRAIN_DIR = "../train-mails"
TEST_DIR = "../test-mails"

word_id = {}
mail_count = 0
class_count = {}

# ---------------------------------------------------------------------------------
# --- PRE PROCESSING STAGE---
# ---------------------------------------------------------------------------------

# Reads all emails and creates a dictionary of the 3000 most common words
# Also creates IDs for each word
def make_dictionary(directory):
    # List of all words in all train mails
    all_words = []
    global mail_count
    # Creates a list of all email files along with their path
    emails = [os.path.join(directory,f) for f in os.listdir(directory)]
    # Iterate through every email and create list of all words
    for mail in emails:
        with open(mail) as m:
            mail_count += 1
            for line in m:
                words = line.split()
                all_words += words

    # Creates a dictionary of all words with their counts
    dictionary = Counter(all_words)


    # DATA CLEANING
    # Remove all punctuations and single letter words
    for item in list(dictionary):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]


    # Working with the 3000 most common words in the dictionary.
    dictionary = dictionary.most_common(3000)

    # Creates a list of words and assigns then an ID.
    # This ID list is later used while feature matrix generation.
    count = 0
    for word in dictionary:
        word_id[word[0]] = count
        count += 1

    return dictionary


# Creates a tabular representation of the dataset
def extract_features(directory):
    email_files = [os.path.join(directory,fi) for fi in os.listdir(directory)]
    features_matrix = np.zeros((len(email_files),3000))
    instance_labels = np.zeros(len(email_files))

    mailId = 0

    for mail in email_files:
        with open(mail) as file:
            for num,line in enumerate(file):
                if num == 2:
                    words = line.split()
                    for word in words:
                        # if d[0] == word:
                        if word in word_id:
                            features_matrix[mailId, word_id[word]] += words.count(word)

        instance_labels[mailId] = 0
        filepathTokens = mail.split('\\')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        if lastToken.startswith("spmsg"):
            instance_labels[mailId] = 1
        mailId = mailId + 1
    return features_matrix, instance_labels


# ---------------------------------------------------------------------------------
# --- SUMMARY/MODEL TRAINING STAGE---
# ---------------------------------------------------------------------------------

# Separates the dataset into dictionary entries based on labels.
# Individual instances are still intact
def partitionByLabels(features_matrix, instance_labels):
    partitioned_features_matrix = {}
    for i in range(len(features_matrix)):

        # Add a new label when encountered during training
        if instance_labels[i] not in partitioned_features_matrix:
            partitioned_features_matrix[instance_labels[i]] = []

        partitioned_features_matrix[instance_labels[i]].append(features_matrix[i])
    return partitioned_features_matrix


def mean(data):
    return sum(data) / float(len(data))


def standard_deviation(data):
    avg = mean(data)
    variance = sum([pow(x - avg, 2) for x in data]) / float(len(data) - 1)
    return math.sqrt(variance)


def summarise_features(dataset):
    summary = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    return summary


# calculates the summary of features for each class
def summarise_class(features_matrix, instance_labels):
    # Feature Matrix is partitioned into a dictionary separated by labels
    partitioned = partitionByLabels(features_matrix, instance_labels)

    # Partitioned Features Matrix is summarised
    features_summary = {}

    for class_label, instances in partitioned.items():
        # The mean and standard deviation of each feature for each label
        features_summary[class_label] = summarise_features(instances)
        class_count[class_label] = len(instances)
    return features_summary


def k_fold_cross_validation(k, features_matrix, instance_labels):
    fold_size = math.floor(len(features_matrix)/k)
    CV_error = []

    for i in range(1,k+1):
        k_start = (int)((i-1)*fold_size)
        k_end = (int)(i*fold_size)
        train_set_features = features_matrix[k_start:k_end]
        train_set_labels = instance_labels[k_start:k_end]

        test_set_features = features_matrix[:k_start] + features_matrix[k_end:]
        test_set_labels = instance_labels[:k_start] + features_matrix[k_end:]

        model = summarise_class(train_set_features , train_set_labels)
        predicted_labels = get_classification(model , test_set_features)

        true_positive = 0
        for i in range(len(test_set_labels)):
            if test_set_labels[i] == predicted_labels[i]:
                true_positive += 1
        mean_square_error = (true_positive^2) / float(len(test_labels))
        CV_error.append(mean_square_error)

    mean_CV_error = CV_error/len(CV_error)
    return mean_CV_error




# ---------------------------------------------------------------------------------
# --- PREDICTION ---
# ---------------------------------------------------------------------------------

# Calculates the probability as per normal distribution
# def calculate_gaussian_probability(x, mean, stdev):
#     exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
#     return (1 / (math.sqrt(2*math.pi) * stdev)) * exp


def calculate_log_gaussian_probability(x, mean, stdev):
    # Calculates the log of the terms as per normal distribution
    exponent = math.pow(x-mean,2)/(2*math.pow(stdev,2))
    ft = math.log(1 / (math.sqrt(2*math.pi) * stdev))
    return ft, exponent


# Calculates probablity of an instance being a part of each class.
def check_class_probabilities(features_summary, input_instance):
    probabilities = {}
    for class_label, class_summary in features_summary.items():

        log_terms = 0
        exponent_terms = 0

        for i, summary in list(enumerate(class_summary)):
            mean , stdev = summary
            x = input_instance[i]
            # If it turns out that some feature has 0 variance, we are going to ignore that.
            if stdev != 0:
                ft , exponent = calculate_log_gaussian_probability(x, mean, stdev)
                log_terms += ft
                exponent_terms += exponent

        probabilities[class_label] = log_terms - exponent_terms + math.log((class_count[class_label]/mail_count))
    return probabilities


# Predicts the class of the function
def classify(features_summary, input_instance):
    probabilities = check_class_probabilities(features_summary , input_instance)

    final_label , maxprob = -1, 1
    for label, prob in probabilities.items():
        # Boundary condition
        if final_label == -1:
            final_label , maxprob = label , prob

        if prob > maxprob:
            final_label, maxprob = label, prob
    return final_label


# ---------------------------------------------------------------------------------
# --- ESTIMATION ---
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



# ---------------------------------------------------------------------------------
# --- DRIVER PROGRAM ---
# ---------------------------------------------------------------------------------

dictionary = make_dictionary(TRAIN_DIR)
print ("reading and processing emails from file.")

features_matrix, labels = extract_features(TRAIN_DIR)
CV_error = k_fold_cross_validation(10,features_matrix, labels)
print ("Cross Validation Error:", CV_error)

# test_feature_matrix, test_labels = extract_features(TEST_DIR)
#
# print("Training model.")
# model = summarise_class(features_matrix , labels)
#
# predicted_labels = get_classification(model , test_feature_matrix)
#
# print("FINISHED classifying. accuracy score : ")
# print (accuracy(test_labels, predicted_labels))
