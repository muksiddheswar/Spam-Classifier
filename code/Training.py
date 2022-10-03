# ---------------------------------------------------------------------------------
# --- SUMMARY/MODEL TRAINING STAGE---
# ---------------------------------------------------------------------------------

import math

class_count = {}


# Separates the dataset into dictionary entries based on labels.
# Individual instances are still intact
def partition_by_labels(features_matrix, instance_labels):
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
    partitioned = partition_by_labels(features_matrix, instance_labels)

    # Partitioned Features Matrix is summarised
    features_summary = {}

    for class_label, instances in partitioned.items():
        # The mean and standard deviation of each feature for each label
        features_summary[class_label] = summarise_features(instances)
        class_count[class_label] = len(instances)
    return features_summary
