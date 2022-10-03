# ---------------------------------------------------------------------------------
# --- PRE PROCESSING STAGE---
# ---------------------------------------------------------------------------------

import os
import numpy as np
from collections import Counter

mail_count = 0
word_id = {}


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