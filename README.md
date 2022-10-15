# Spam-Classifier

This is an implementation of the Naive Bayes Classification technique as a Spam Classifier.    
Coded from the scratch in Python.

### Classifier Logic:
- Scan all mails to get the top 3000 most used words.
- Convert each mail into a feature matrix based on this.
- Calculate the summary of these top 3000 words for each label.
- For a given mail calculate the log gaussian probability for each class.
- Label with the highest probability wins.

---

Each version has it's own branch.   
Master is the latest version. (And possibly under development.)    

~~_Working on making this project more generic._~~   
No more changes to the actual classifier logic will be done.    
Future plans for this project include creation of a Flask based APIs that will:
1. Trigger creation of the class summary.
2. Read test emails from a specified location.

## Version: V02_02    
- Add docstring and comments.
- Optimised variable usage.
- Fixed bugs.
- Better logging.

![image](https://user-images.githubusercontent.com/17706548/195998689-ddf8480f-d47f-4615-99c3-aa35f03fe411.png)


## Version: V02_01    

![image](https://user-images.githubusercontent.com/17706548/194677163-c0e0f80c-7a59-4b44-b6d5-da15e90ec888.png)


- Currently classification works in the specified set of mails.
- Modularised source code into separate files.
- Removed hard coded paths. (config.txt)

## Version: V01_01:
- Single script for preprocessing, training, testing. 
- Simple implementation I did as a part of class project during my masters degree.

___


#### Reference:
https://github.com/savanpatel    
https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
