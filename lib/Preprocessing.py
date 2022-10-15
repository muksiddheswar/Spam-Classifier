# ---------------------------------------------------------------------------------
# --- PRE PROCESSING STAGE---
# ---------------------------------------------------------------------------------

import os
from typing import Tuple, List, Any, Dict

import numpy as np
from collections import Counter


def quantify_mails(directory: str):
    """
    Reads all emails and creates a list of tuples of the 3000 most common words.
    Also creates IDs for each word.

    :param directory: str
    :return: list[tuple], dict[str, int]
    """
    # List of all words in all train mails
    all_words = []
    mail_count = 0
    word_id = {}

    # Creates a list of all email files along with their path
    emails = [os.path.join(directory, f) for f in os.listdir(directory)]

    # Iterate through every email and create list of all words
    for mail in emails:
        with open(mail) as m:
            mail_count += 1
            for line in m:
                words = line.split()
                all_words += words

    # Creates a dictionary (Counter) of all words with their counts
    top_word_counts = Counter(all_words)

    # Remove all punctuations and single letter words
    for item in list(top_word_counts):
        if not item.isalpha():
            del top_word_counts[item]
        elif len(item) == 1:
            del top_word_counts[item]

    # Working with the 3000 most common words in the dictionary.
    # most_common return a list of tuples
    top_word_counts = top_word_counts.most_common(3000)

    for index, word in enumerate(list(top_word_counts)):
        word_id[word[0]] = index + 1

    # f = open("mail_count.txt", "w")
    # f.write(str(mail_count))
    # f.close()

    return top_word_counts, word_id


# Creates a tabular representation of the dataset
def extract_features(directory):
    top_word_counts, word_id = quantify_mails(directory)
    email_files = [os.path.join(directory, fi) for fi in os.listdir(directory)]
    features_matrix = np.zeros((len(email_files), 3000))
    instance_labels = np.zeros(len(email_files))

    print("Calculating feature matrix for " + os.path.basename(directory) + " ...", end='')

    mail_id = 0

    for mail in email_files:
        if (mail_id % 100) == 0:
            print(".", end='')

        with open(mail) as file:
            # The files are in a format such that the main content in the line 3 of the file.
            for num, line in enumerate(file):
                if num == 2:
                    words = line.split()
                    for word in words:
                        if word in word_id:
                            features_matrix[mail_id, word_id[word]-1] += words.count(word)

        filepath_tokens = mail.split('\\')
        last_token = filepath_tokens[len(filepath_tokens) - 1]
        if last_token.startswith("spmsg"):
            instance_labels[mail_id] = 1
        else:
            instance_labels[mail_id] = 0

        mail_id = mail_id + 1

    print("\nSuccessfully calculated feature matrix for " + os.path.basename(directory) + ".")
    return features_matrix, instance_labels


TRAIN_DIR = "../train-mails"
# top_word_counts_, word_id_ = quantify_mails(TRAIN_DIR)
# print(top_word_counts_)
# print(len(top_word_counts_))
# print(type(top_word_counts_))
# print(word_id_)
# print(type(word_id_))
# print(len(word_id_))


# features_matrix_, instance_labels_ = extract_features(TRAIN_DIR)
# print(features_matrix_.shape)
# print(instance_labels_.shape)