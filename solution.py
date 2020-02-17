import pandas as pd

train_set = 'UNSW_NB15_training-set.csv'
test_set = 'UNSW_NB15_testing-set.csv'

training = pd.read_csv(train_set, index_col='id')
testing = pd.read_csv(test_set, index_col='id')
