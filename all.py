import csv
import numpy as np
import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

training = pd.read_csv('UNSW_NB15_training-set.csv')
testing = pd.read_csv('UNSW_NB15_testing-set.csv')

# This mess is where you can play with different features
train_x = training.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'swin', 'dload']]
test_x = testing.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'swin', 'dload']]

#train_x = training.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'swin', 'dload']]
#test_x = testing.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'swin', 'dload']]
#test_x = testing.loc[:, ['sttl', 'rate', 'dur']]
#train_x = training.loc[:, ['sttl', 'rate', 'dur']]


# These will pretty much stay the same
train_y = training.loc[:,['label']]
test_y = testing.loc[:,['label']]


def tree():
    print("--------------------------------")
    print("Decision Tree")
    # Just chose an arbitrarily high depth
    model = DecisionTreeClassifier(max_depth=9000)
    model.max_iter = 20000
    model.fit(train_x, train_y.values.ravel())

    # Report how successful model was
    print("Test : %s" % model.score(test_x, test_y.values.ravel()))
    print("Train: %s" % model.score(train_x, train_y.values.ravel()))
    predictions = model.predict(test_x)

    # Confusion Matrix testing data
    print("\nCM - Testing Model")
    cm = metrics.confusion_matrix(test_y, predictions)
    print(cm)

    # Confusion Matrix training data
    trainpred = model.predict(train_x)
    tcm = metrics.confusion_matrix(train_y, trainpred)
    print("\nCM - Training Model")
    print(tcm)

    # Do report writing
    # Write Preditions to CSV file
    report_csv('Result-DecisionTree.csv', predictions)
    report_matrix('Decision Tree', cm, model.score(train_x, train_y.values.ravel()), tcm, model.score(test_x, test_y.values.ravel()))

def KNN():
    print("\n--------------------------------")
    print("KNN")
    model = KNeighborsClassifier(n_neighbors=3)
    model.max_iter = 20000
    model.fit(train_x, train_y.values.ravel())

    # Report how successful model was
    print("Test : %s" % model.score(test_x, test_y.values.ravel()))
    print("Train: %s" % model.score(train_x, train_y.values.ravel()))
    predictions = model.predict(test_x)

    # Confusion Matrix testing data
    print("\nCM - Testing Model")
    cm = metrics.confusion_matrix(test_y, predictions)
    print(cm)

    # Confusion Matrix training data
    trainpred = model.predict(train_x)
    tcm = metrics.confusion_matrix(train_y, trainpred)
    print("\nCM - Training Model")
    print(tcm)

    # Do report writing
    # Write Preditions to CSV file
    report_csv('Result-KNN.csv', predictions)
    report_matrix('KNN', cm, model.score(train_x, train_y.values.ravel()), tcm, model.score(test_x, test_y.values.ravel()))

def LR():
    print("\n--------------------------------")
    print("Logistic Regression")
    model = LogisticRegression()
    model.max_iter = 20000
    model.fit(train_x, train_y.values.ravel())

    # Report how successful model was
    print("Test : %s" % model.score(test_x, test_y.values.ravel()))
    print("Train: %s" % model.score(train_x, train_y.values.ravel()))
    predictions = model.predict(test_x)

    # Confusion Matrix testing data
    print("\nCM - Testing Model")
    cm = metrics.confusion_matrix(test_y, predictions)
    print(cm)

    # Confusion Matrix training data
    trainpred = model.predict(train_x)
    tcm = metrics.confusion_matrix(train_y, trainpred)
    print("\nCM - Training Model")
    print(tcm)

    # Do report writing
    # Write Preditions to CSV file
    report_csv('Result-LogisticRegression.csv', predictions)
    report_matrix('Logistic Regression', cm, model.score(train_x, train_y.values.ravel()), tcm, model.score(test_x, test_y.values.ravel()))

def report_csv(csvname, data):
    with open('./reports/'+csvname, mode='w') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(data)

def report_matrix(model, cm, atrain, tcm, atest):
    with open('./reports/final-report.txt', mode='a+') as file:
        file.write("************************\n")
        file.write(model + "\n")
        file.write("------------------------\n")
        file.write("Train Accuracy: %s\n" % atrain)
        file.write(str(tcm))
        file.write("\n\n")

        file.write("Test Accuracy:  %s\n" % atest)
        file.write(str(cm))
        file.write("\n\n")

if __name__ == "__main__":
    # Run the models, it was done this way to make selective execution easier :)
    tree()
    KNN()
    LR()
