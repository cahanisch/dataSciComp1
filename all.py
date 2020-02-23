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

#print(training.head())
#print(training.corr())

train_x = training.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'swin', 'dload']]
#train_x = training.loc[:, ['sttl', 'rate', 'dur']]
#train_x = training.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'swin', 'dload']]
test_x = testing.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'swin', 'dload']]
#test_x = testing.loc[:, ['sttl', 'rate', 'dur']]
#test_x = testing.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'swin', 'dload']]
train_y = training.loc[:,['label']]
test_y = testing.loc[:,['label']]

def tree():
    print("--------------------------------")
    print("Decision Tree")
    model = DecisionTreeClassifier(max_depth=9000) # idk what to put for max depth, it doesn't get any better with higher nums
    model.max_iter = 20000
    model.fit(train_x, train_y.values.ravel())

    # Report how successful model was
    print("Test : %s" % model.score(test_x, test_y.values.ravel()))
    print("Train: %s" % model.score(train_x, train_y.values.ravel()))
    predictions = model.predict(test_x)

    trainpred = model.predict(train_x)
    tcm = metrics.confusion_matrix(train_y, trainpred)
    print("\nCM - Training Model")
    print(tcm)

    print("\nCM - Testing Model")
    cm = metrics.confusion_matrix(test_y, predictions)
    print(cm)

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

    trainpred = model.predict(train_x)
    tcm = metrics.confusion_matrix(train_y, trainpred)
    print("\nCM - Training Model")
    print(tcm)

    print("\nCM - Testing Model")

    cm = metrics.confusion_matrix(test_y, predictions)
    print(cm)

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

    trainpred = model.predict(train_x)
    tcm = metrics.confusion_matrix(train_y, trainpred)
    print("\nCM - Training Model")
    print(tcm)

    print("\nCM - Testing Model")

    cm = metrics.confusion_matrix(test_y, predictions)
    print(cm)

if __name__ == "__main__":
    tree()
    KNN()
    LR()
