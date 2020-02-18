import numpy as np
import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn import metrics
#from collections import Counter

training = pd.read_csv('UNSW_NB15_training-set.csv')
testing = pd.read_csv('UNSW_NB15_testing-set.csv')
#print(training.head())
#print(training.corr())
train_x = training.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'swin', 'dload']]
train_y = training.loc[:,['label']]
test_x = testing.loc[:, ['sttl', 'dmean', 'rate', 'dwin', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'swin', 'dload']]
test_y = testing.loc[:,['label']]
logisticRegr = LogisticRegression()
logisticRegr.max_iter = 20000
logisticRegr.fit(train_x, train_y.values.ravel())
print(logisticRegr.score(test_x, test_y.values.ravel()))
predictions = logisticRegr.predict(test_x)
cm = metrics.confusion_matrix(test_y, predictions)
print(cm)