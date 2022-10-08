from lib.Naive_Bayes_Classifier import naive_bayes_classifier
from lib.cleanup import cleanup
import os


def main():
    if os.path.isfile("config.txt"):

        f = open("config.txt", "r")
        line = f.readline().rstrip('\n')
        f.close()

        datafolder = os.path.normcase(line.split("=")[1])

        train_dir = os.path.join(datafolder, "train-mails")
        test_dir = os.path.join(datafolder, "test-mails")

    else:
        train_dir = os.path.join(os.path.curdir, "train-mails")
        test_dir = os.path.join(os.path.curdir, "test-mails");

    naive_bayes_classifier(train_dir, test_dir)
    cleanup(os.path.abspath(os.getcwd()))

    return 0


if __name__ == "__main__":
    main()
