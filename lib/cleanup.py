import os


def cleanup(root_folder):
    """

    :rtype: None
    """
    root_folder = os.path.normcase(root_folder)
    count_files = [os.path.join(root_folder, "lib", "class_count.txt"),
                   os.path.join(root_folder, "lib", "mail_count.txt"),
                   os.path.join(root_folder, "class_count.txt"),
                   os.path.join(root_folder, "mail_count.txt")]

    print()
    for file in count_files:
        if os.path.isfile(file):
            os.remove(file)

    print("Cleanup Successful.")
