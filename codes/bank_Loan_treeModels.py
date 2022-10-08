from sys import prefix
import warnings
from sklearn import preprocessing
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import timeit

def importance_features_top(model_str, model, x_train): # importance features
    print('*'*10," Print importance features ",'*'*10)
    feature_importances_ = model.feature_importances_
    feature_names = x_train.columns
    importance_col = pd.DataFrame([*zip(feature_names, feature_importances_)], 
                                  columns=['a', 'b'])
    importance_col_desc = importance_col.sort_values(by='b', ascending=False)
    print(importance_col_desc.iloc[:10, :])

def xgboost_model(x_train,y_train,num_class): # customize xgboost
    xgboost_clf = XGBClassifier(min_child_weight=1,max_depth=50, n_estimators=100,
                                objective='binary:logistic',learning_rate=0.001,eval_metric='auc')
    print("-" * 60)
    print("xgboost Model:", xgboost_clf)
    xgboost_clf.fit(x_train, y_train)
    importance_features_top('xgboost', xgboost_clf, x_train)
    # joblib.dump(xgboost_clf, './Models/XGBoost_model_v1.0') # save model
    return xgboost_clf

def print_metrics(y_true, y_pre):
    print("Print Accuracy, Precision, Recall and F1")
    print(classification_report(y_true, y_pre, digits=4))
    ac = round(accuracy_score(y_true, y_pre), 4)
    p = round(precision_score(y_true, y_pre, average='macro'), 4)
    r = round(recall_score(y_true, y_pre, average='macro'), 4)
    f1 = round(f1_score(y_true, y_pre, average='macro'), 4)
    print("Accuracy: {}, Precision: {}, Recall: {}, F1: {} ".format(ac, p, r, f1))

target = 'personal_loan'

selected_variables = ['age','experience','income','family','ccavg','education','mortgage','securities_account','cd_account','online','creditcard','personal_loan']
categorical_variables = ['family','education','securities_account','cd_account','online','creditcard','personal_loan']
categorical_variables.append(categorical_variables.pop(categorical_variables.index(target))) # move the target column to the end
continuous_variables = [col for col in selected_variables if col not in categorical_variables]

orig_data = pd.read_csv('./sourceData/bank_Loan.csv', usecols=selected_variables)

target_num = orig_data[target].nunique()
print('Output dimension: ', target_num)
print('Classes to predict:\n', orig_data[target].value_counts())
# preprocessing, onehot, minmaxscale
le = preprocessing.LabelEncoder()
for col in categorical_variables:
    orig_data[col]=le.fit_transform(orig_data[col])
encoded_data = orig_data
scaler = MinMaxScaler(feature_range=(-1, 1))
encoded_data[continuous_variables] = scaler.fit_transform(encoded_data[continuous_variables])

print(encoded_data.head())
print('encoded data shape: ', encoded_data.shape)
trainSet = encoded_data.sample(frac=0.8, random_state=64)
testSet = encoded_data.drop(trainSet.index)
train_X = trainSet.iloc[:, :-1]
train_y = trainSet.iloc[:, -1:]
print(train_y.head())
test_X = testSet.iloc[:, :-1]
test_y = testSet.iloc[:, -1:]
print('train_X', train_X.shape)
print('test_X', test_X.shape)
print('train_y', train_y.shape)
print('test_y', test_y.shape)
# hyperparameters of finmodel
tic = timeit.default_timer()

print(le.classes_)

def runModel():
    xgboost_clf = xgboost_model(train_X, train_y,target_num)
    pre_y_test = xgboost_clf.predict(test_X)

    print_metrics(test_y, pre_y_test)
    toc=timeit.default_timer()
    print('Total Training Time is', toc-tic)

runModel()
