# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:38:07 2020

@author: LMurphy
"""


### MODULES
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score




#### READ IN THE DATA

modelsamp = pd.read_csv('C:/Users/lmurphy/OneDrive - Three/Desktop/Data Science MSc/Dissertation and Project/Project/Data/Sample_prepped.csv', sep=',')

modelsamp.info()
modelsamp.describe()
modelsamp.head(5)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



# ### Data Partitioning

# Split out features and target into X and y
X=modelsamp.drop('target', axis=1)
y=modelsamp.target

# Split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# create y_train data for PLS approach
import numpy as np
N = 10
a = y_train
b = 1-a
c = a,b

pls_y_train = np.asarray(c).transpose()


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# ### Recursive Feature Elimination (RFE)

from sklearn.feature_selection import RFE

# RFE(estimator, *, n_features_to_select=None, step=1, verbose=0)

# I'm choosing 30 inputs as a start
model = LogisticRegression(max_iter=50)
rfe = RFE(model, 50)

# we have selected the best 20 features
rfe = rfe.fit(X_train, y_train)
print (rfe.support_)
print (rfe.ranking_)


rfe_rankinglist = rfe.ranking_.tolist()
selected_columns = [n for im, n in enumerate(X_train) if rfe_rankinglist[im]==1]
print (selected_columns)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### REGRESSION APPROACHES ####


###############################################################################

## Logistic regression - UNTUNED
# Using RFE selected features

LR = LogisticRegression(max_iter = 500)

LR.fit(X_train[selected_columns],y_train)

LR.score(X_train[selected_columns], y_train)
LR.score(X_test[selected_columns], y_test)



# ROC Curve for Model Assessment
# roc curve and auc
from sklearn import metrics

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = LR.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Logictic regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()




### Lift chart

# SVC is whatever your model classifier would be
predicted_probas = LR.predict_proba(X_test[selected_columns])

test=pd.DataFrame(data=predicted_probas[0:,0:])

y_test.to_frame()

final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)

final.columns = ['nopredict','predict','actual']

# Sort on prediction (descending)
# Add row ids 
# Add decile 
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)

# Check the count by decile
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()

#create gains table
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains

#add metrics to the gains table
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)

#gains.plot(y='lift', label='Lift')
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Logistic Regression Model Lift')
plt.show()



###############################################################################


## Logistic regression - TUNED
# Using RFE selected features

from sklearn.model_selection import GridSearchCV

grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}

LR2 = LogisticRegression(max_iter = 500)

grid_LR = GridSearchCV(LR2, param_grid = grid_values)

grid_LR.fit(X_train[selected_columns],y_train)

grid_LR.score(X_train[selected_columns], y_train)
grid_LR.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = grid_LR.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Tuned Receiver Operating Characteristic - Logictic regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()


# Lift chart
predicted_probas = grid_LR.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned Logistic Regression Model Lift')
plt.show()



###############################################################################

## Partial Least Squares regression - UNTUNED

from sklearn.cross_decomposition import PLSRegression
#from sklearn.preprocessing import StandardScaler


s#caler = StandardScaler()

# Fit only to the training data
#scaler.fit(X_train)

# Now apply the transformations to the data:
    
#X_train_scaler = scaler.transform(X_train)
#X_test_scaler = scaler.transform(X_test)

#PLS = PLSRegression(max_iter = 500, n_components=12, scale=True)
PLS = PLSRegression(max_iter = 200, n_components=15, scale=True)


PLS.fit(X_train, y_train)

PLS.score(X_train, y_train)
PLS.score(X_test, y_test)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### DECISION TREE APPROACHES ####


###############################################################################

## Decision tree - UNTUNED
 

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

DT = DecisionTreeClassifier(max_depth = 6)

DT.fit(X_train[selected_columns], y_train)

DT.score(X_train[selected_columns], y_train)
DT.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = DT.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Decision Tree')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = DT.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Decision Tree Model Lift')
plt.show()






###############################################################################

## Decision tree - TUNED

grid_values = {"criterion": ['gini', 'entropy'],
               "max_depth":range(1,10),
               "min_samples_split":range(1,10),
               "min_samples_leaf":range(1,5)}

grid_DT = GridSearchCV(DT,
                       param_grid = grid_values,
                       verbose = 1,
                       n_jobs = -1)

grid_DT.fit(X_train[selected_columns], y_train)

grid_DT.score(X_train[selected_columns], y_train)
grid_DT.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = grid_DT.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned Decision Tree')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = grid_DT.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned Decision Tree Model Lift')
plt.show()


###############################################################################

## Random forest - *** UNTUNED
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators = 1000, random_state = 42, max_depth =6)

# train the model on training data
RF.fit(X_train[selected_columns], y_train)

RF.score(X_train[selected_columns], y_train)
RF.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = RF.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Random Forest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = RF.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Random Forest Model Lift')
plt.show()


###############################################################################

## Random forest - ***TUNED

RF2 = RandomForestClassifier()

param_grid = { "n_estimators"      : [250, 300],
           "criterion"         : ["gini", "entropy"],
           "max_features"      : [3, 5],
           "max_depth"         : [2, 4],
           "min_samples_split" : [2, 4] ,
           "bootstrap": [True, False]}

grid_RF = GridSearchCV(RF2, param_grid)

grid_RF.fit(X_train[selected_columns], y_train)

grid_RF.score(X_train[selected_columns], y_train)
grid_RF.score(X_test[selected_columns], y_test)



# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = grid_RF.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned Random Forest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = grid_RF.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned Random Forest Model Lift')
plt.show()



###############################################################################


# XG Gradient Boost model - UNTUNED

pip install xgboost

from xgboost import XGBClassifier
from xgboost import plot_importance

XGB = XGBClassifier(max_depth=5)
XGB.fit(X_train[selected_columns], y_train)

# XGBoost model performance
XGB.score(X_train[selected_columns], y_train)
XGB.score(X_test[selected_columns], y_test)



# Use all available inputs
XGB.fit(X_train, y_train)

XGB.score(X_train, y_train)
XGB.score(X_test, y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = XGB.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - XG Boost')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = XGB.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('XG Boost Model Lift')
plt.show()




###############################################################################


# XG Gradient Boost model ***TUNED

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_importance

#param_grid = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
# "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
# "min_child_weight" : [ 1, 3, 5, 7 ],
# "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
# "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

param_grid = {"learning_rate"    : [0.05, 0.15, 0.25] ,
 "max_depth"        : [ 5, 10],
 "min_child_weight" : [ 1,  ],
 "gamma"            : [ 0.0, 0.1, 0.3],
 "colsample_bytree" : [ 0.3, 0.5] }



#param_grid = {
#    # Parameters that we are going to tune.
#    'max_depth':6,
#    'min_child_weight': 1,
#    'eta':.3,
#    'subsample': 1,
#    'colsample_bytree': 1,
#    # Other parameters
#    'objective':'reg:linear',
#}

XGB_HYP = GridSearchCV(XGB, param_grid)

XGB_HYP.fit(X_train[selected_columns], y_train)

# XGBoost model performance
XGB_HYP.score(X_train[selected_columns], y_train)
XGB_HYP.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = XGB_HYP.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned XG Boost')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = XGB_HYP.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned XG Boost Model Lift')
plt.show()




###############################################################################


# Cross validation on Tuned XGBoost

param_grid = {"learning_rate"    : [0.05, 0.15, 0.25] ,
 "max_depth"        : [ 5, 10],
 "min_child_weight" : [ 1, 3, 5],
 "gamma"            : [ 0.0, 0.1, 0.3],
 "colsample_bytree" : [ 0.3, 0.5] }

XGB = XGBClassifier()

XGB_HYP = GridSearchCV(XGB, param_grid, cv=10)
XGB_HYP.fit(X_train[selected_columns], y_train)

# XGBoost model performance
XGB_HYP.score(X_train[selected_columns], y_train)
XGB_HYP.score(X_test[selected_columns], y_test)



# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = XGB_HYP.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned XG Boost')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = XGB_HYP.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned XG Boost Model Lift')
plt.show()



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### LIGHT GBM ####

pip install lightgbm

import lightgbm
from lightgbm import LGBMClassifier

LGBM = LGBMClassifier()

LGBM.fit(X_train[selected_columns], y_train)

# LGBM model performance
LGBM.score(X_train[selected_columns], y_train)
LGBM.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = LGBM.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - LGBM')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = LGBM.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('LGBM Model Lift')
plt.show()


###############################################################################


# LGBM ***TUNED

from sklearn.model_selection import GridSearchCV


param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }



LGBM = LGBMClassifier()

LGBM_TUNED = GridSearchCV(LGBM, param_grid)

LGBM_TUNED.fit(X_train[selected_columns], y_train)

# LGBM model performance
LGBM_TUNED.score(X_train[selected_columns], y_train)
LGBM_TUNED.score(X_test[selected_columns], y_test)


print(LGBM_TUNED.get_params())



# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = LGBM_TUNED.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned LGBM')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = LGBM_TUNED.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned LGBM Lift')
plt.show()


# summarize results
grid_result = LGBM_TUNED.fit(X_train[selected_columns], y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Feature importance
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_importance

# Feature importance
# Doesn't run for grid search resulst
LGBM_feature_importance = LGBM.feature_importances_

# Get feature names
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X_train.columns, LGBM.feature_importances_):
    feats[feature] = importance #add the name/value pair 

# Plot top features
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
top_20 = importances.sort_values('Gini-importance', ascending = False)[:20]
top_20.sort_values(by='Gini-importance').plot(kind='barh') 



###############################################################################


# LGBM NO RFE

from sklearn.model_selection import GridSearchCV


#param_grid = {
#    'num_leaves': [31, 127],
#    'reg_alpha': [0.1, 0.5],
#    'min_data_in_leaf': [30, 50, 100, 300, 400],
#    'lambda_l1': [0, 1, 1.5],
#    'lambda_l2': [0, 1]
#    }


LGBM_v2 = LGBMClassifier()

#LGBM_TUNED = GridSearchCV(LGBM, param_grid)

LGBM_v2.fit(X_train, y_train)

# LGBM model performance
LGBM_v2.score(X_train, y_train)
LGBM_v2.score(X_test, y_test)



# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = LGBM_v2.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - LGBM model')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = LGBM_v2.predict_proba(X_test)
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('LGBM Lift')
plt.show()


# summarize results
#grid_result = LGBM_v2.fit(X_train[selected_columns], y_train)

#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Feature importance
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_importance

# Feature importance
# Doesn't run for grid search resulst
LGBM_feature_importance_v2 = LGBM_v2.feature_importances_

# Get feature names
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X_train.columns, LGBM_v2.feature_importances_):
    feats[feature] = importance #add the name/value pair 

# Plot top features
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
top_20 = importances.sort_values('Gini-importance', ascending = False)[:20]
top_20.sort_values(by='Gini-importance').plot(kind='barh') 



### TUNED
from sklearn.model_selection import GridSearchCV


param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }


LGBM = LGBMClassifier()

LGBM_TUNED_v2 = GridSearchCV(LGBM, param_grid)

LGBM_TUNED_v2.fit(X_train, y_train)

# LGBM model performance
LGBM_TUNED_v2.score(X_train, y_train)
LGBM_TUNED_v2.score(X_test, y_test)



# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = LGBM_TUNED_v2.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned LGBM')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = LGBM_TUNED_v2.predict_proba(X_test)
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned LGBM Lift')
plt.show()




### RANDOMISED SEARCH

from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier


LGBM = LGBMClassifier()

clf = LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=100,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

gs.fit(X_train, y_train,)

#  model performance
gs.score(X_train, y_train)
gs.score(X_test, y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = gs.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Random-search tuned LGBM')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = gs.predict_proba(X_test)
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title(' Random-search tuned LGBM Lift')
plt.show()



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### NEURAL NETWORK APPROACH- UNTUNED ####

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

NN = MLPClassifier(hidden_layer_sizes=(30,30,30))

NN.fit(X_train[selected_columns],y_train)

NN.score(X_train[selected_columns], y_train)
NN.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = NN.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Neural Network')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = NN.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Neural Network Model Lift')
plt.show()



###############################################################################

#### NEURAL NETWORK APPROACH - TUNED ####

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


NN2 = MLPClassifier(max_iter=100)

param_grid = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}




grid_NN = GridSearchCV(NN2, param_grid, n_jobs=-1)

grid_NN.fit(X_train[selected_columns],y_train)

grid_NN.score(X_train[selected_columns], y_train)
.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = grid_NN.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned Neural Network')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = grid_NN.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned Neural Network Model Lift')
plt.show()




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### SUPPORT VECTOR MACHINE APPROACH - UNTUNED ####

from sklearn import preprocessing, svm
from sklearn.svm import SVC

# SVC (*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, 
#      cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False,
#      random_state=None)

SVC = SVC(C = 1.0, kernel = "linear")

SVC.fit(X_train,y_train)

SVC.score(X_train, y_train)
SVC.score(X_test, y_test)


###############################################################################


#### SUPPORT VECTOR MACHINE APPROACH - TUNED ####


param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### KNN - UNTUNED ####

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 200)

KNN.fit(X_train[selected_columns], y_train)

KNN.score(X_train[selected_columns], y_train)
KNN.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = KNN.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - K Nearest Neighbour')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = KNN.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('K Nearest Neighbour Model Lift')
plt.show()



###############################################################################

#### KNN - TUNED ####
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(5, 250),
              'weights': ["uniform", "distance"]}

knn2 = KNeighborsClassifier()

#use gridsearch to test all values for n_neighbors
grid_KNN = GridSearchCV(knn2, param_grid)

grid_KNN.fit(X_train[selected_columns], y_train)

grid_KNN.score(X_train[selected_columns], y_train)
grid_KNN.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = grid_KNN.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Tuned K Nearest Neighbour')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = grid_KNN.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned K Nearest Neighbour Model Lift')
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#### ENSEMBLE APPROACH ####


# Ensemble model

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.tree import DecisionTreeClassifier

#LR = LogisticRegression()
#NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#DT = DecisionTreeClassifier(random_state=0)

models = []
models.append(('LR', LR))
models.append(('NN', NN))
models.append(('DT', DT))

estimators = [m for m in models]
ensemble_orig = VotingClassifier(estimators, voting='soft')

# ensemble model ift
ensemble_orig.fit(X_train[selected_columns], y_train)

# ensemble model performance
ensemble_orig.score(X_train[selected_columns], y_train)
ensemble_orig.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = ensemble_orig.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Ensemble')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = ensemble_orig.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Ensemble Model Lift')
plt.show()


###############################################################################


#### ENSEMBLE APPROACH ON TUNED MODELS ####


# Ensemble model

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.tree import DecisionTreeClassifier

#LR = LogisticRegression()
#NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#DT = DecisionTreeClassifier(random_state=0)

models = []
models.append(('grid_LR', grid_LR))
models.append(('grid_NN', grid_NN))
models.append(('grid_DT', grid_DT))

estimators = [m for m in models]
ensemble_grid = VotingClassifier(estimators, voting='soft')

# ensemble model ift
ensemble_grid.fit(X_train[selected_columns], y_train)

# ensemble model performance
ensemble_grid.score(X_train[selected_columns], y_train)
ensemble_grid.score(X_test[selected_columns], y_test)


# ROC 
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = ensemble_grid.predict_proba(X_test[selected_columns])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Ensemble Tuned')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate (Specificity)')
plt.show()



# Lift chart
predicted_probas = ensemble_grid.predict_proba(X_test[selected_columns])
test=pd.DataFrame(data=predicted_probas[0:,0:])
y_test.to_frame()
final = pd.concat([test.reset_index(drop='True'), y_test.reset_index(drop='True')], axis=1)
final.columns = ['nopredict','predict','actual']
data= final.sort_values(by='predict',ascending=False)
data['row_id'] = range(0,0+len(data))
data['decile'] = ( data['row_id'] / (len(data)/10) ).astype(int)
data.loc[data['decile'] == 10]=9
data['decile'].value_counts()
gains = data.groupby('decile')['actual'].agg(['count','sum'])
gains.columns = ['count','actual']
gains
gains['non_actual'] = gains['count'] - gains['actual']
gains['cum_count'] = gains['count'].cumsum()
gains['cum_actual'] = gains['actual'].cumsum()
gains['cum_non_actual'] = gains['non_actual'].cumsum()
gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
gains['if_random'] = np.max(gains['cum_actual']) /10 
gains['if_random'] = gains['if_random'].cumsum()
gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual'] ) * 100
gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
gains = pd.DataFrame(gains)
gains.head(100)
gains.plot.bar(y='lift', label='Lift', colormap='Paired')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(np.arange(0, 10, 1)) 
plt.title('Tuned Ensemble Model Lift')
plt.show()

    
    
#############################################################################
#############################################################################
#########################  E-N-D  O-F  P-R-O-G-R-A-M ########################
#############################################################################
#############################################################################
