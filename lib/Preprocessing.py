# ---------------------------------------------------------------------------------
# --- PRE PROCESSING STAGE---
# ---------------------------------------------------------------------------------

import os
from typing import Tuple, List, Any, Dict

import numpy as np
from collections import Counter


def create_word_database(directory: str):
    """
    Reads all emails and creates a list of tuples of the 3000 most common words.
    Also creates IDs for each of the top 3000 word.

    :param directory: str
    :return: list[tuple], dict[str, int]
    """
    all_words = []
    mail_count = 0
    top_word_id = {}

    print("Create DB: Calculating top 3000 word database.")

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
    top_words = Counter(all_words)

    # Remove all punctuations and single letter words
    for item in list(top_words):
        if not item.isalpha():
            del top_words[item]
        elif len(item) == 1:
            del top_words[item]

    # Working with the 3000 most common words in the dictionary.
    # most_common return a list of tuples
    top_words = top_words.most_common(3000)

    for index, word in enumerate(list(top_words)):
        top_word_id[word[0]] = index + 1

    # f = open("mail_count.txt", "w")
    # f.write(str(mail_count))
    # f.close()

    print("Create DB: Successfully completed top database creation.")
    print("Create DB: Files scanned: " + mail_count.__str__() + ".")

    return top_words, top_word_id


def extract_features(directory):
    """
    Creates a tabular representation of the dataset.
    Reads each file and converts it into a feature matrix of length 3000.
    Calculates count of each of the top 3000 words in the DB.

    :type directory: str

    """
    top_words, top_word_id = create_word_database(directory)
    email_files = [os.path.join(directory, fi) for fi in os.listdir(directory)]
    email_features_matrix = np.zeros((len(email_files), 3000))
    instance_labels = np.zeros(len(email_files))

    print("Create FT: Calculating feature matrix for " + os.path.basename(directory) + " ...", end='')

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
                        if word in top_word_id:
                            email_features_matrix[mail_id, top_word_id[word] - 1] += words.count(word)

        filepath_tokens = mail.split('\\')
        last_token = filepath_tokens[len(filepath_tokens) - 1]
        if last_token.startswith("spmsg"):
            instance_labels[mail_id] = 1
        else:
            instance_labels[mail_id] = 0

        mail_id = mail_id + 1

    print("\nCreate FT: Successfully calculated feature matrix for " + os.path.basename(directory) + ".")
    return email_features_matrix, instance_labels


# TRAIN_DIR = "../train-mails"
# top_word_counts_, word_id_ = quantify_mails(TRAIN_DIR)
# features_matrix_, instance_labels_ = extract_features(TRAIN_DIR)
# print(features_matrix_.shape)
# print(instance_labels_.shape)
