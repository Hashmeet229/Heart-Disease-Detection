import numpy as np
import pandas as pd
#import libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score

import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("heart.csv")
data = np.array(data)

X=data[1:,:-1]
y=data[1:,-1]
print(X)
print(y)
#y = y.astype('int')
#X = X.astype('int')
# print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
inputt=[float(x) for x in "63,1,3,145,233,1,0,150,0".split(',')]
final=[np.array(inputt)]

b=log_reg.predict_proba(final)
print(b)

y_pred=log_reg.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))

pickle.dump(log_reg, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))