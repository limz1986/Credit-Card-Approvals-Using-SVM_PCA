# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:46:21 2022

@author: 65904
"""

from numba import jit, cuda
import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, brier_score_loss


df = pd.read_csv (r'C:/Users/65904/Desktop/Machine Learning/Datasets/Credit Card_Approval_dataset.csv')

#EDA
df.dtypes
df.describe()
df['CreditScore'].unique()

#Outlier/Cleaning
credit_outlier_filter = df['CreditScore'] < 0
df.loc[credit_outlier_filter,'CreditScore'] =  df.loc[credit_outlier_filter,'CreditScore'] * -1

for col in ['Income','Age','CreditScore']:
    df.boxplot(column=[col])
    plt.show()

#X,y Split
X = df.drop('Approved', axis=1).copy() 
X.head() 
y = df['Approved'].copy()
y.head()

#One Hot Encoding
X_encoded = pd.get_dummies(X, columns=['Industry',
                                       'Ethnicity',
                                       'Citizen'
                                       ])
X_encoded.head()
X_encoded.dtypes

#Stratification
sum(y)/len(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

#XGBoost
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', 
                            eval_metric="logloss",
                            seed=42, 
                            use_label_encoder=False)

clf_xgb.fit(X_train, 
            y_train)
            
            
plot_confusion_matrix(clf_xgb, X_test, y_test, display_labels=["Not Approved", "Approved"
                                                               ])



params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': range(50, 250, 50),
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0, 1.0, 10.0, 100.0]
}

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric="logloss", seed=42, use_label_encoder=False),
    param_grid=params,
    scoring = 'roc_auc',
    # subsample=0.9, 
    verbose=0,
    n_jobs = 10,
    cv = 5
)

optimal_params.fit(X_train, y_train)
print(optimal_params.best_params_)

clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        eval_metric="logloss", 
                        gamma=0.2,
                        colsample_bytree = 0.5, 
                        learning_rate=0.1,
                        max_depth=3, 
                        n_estimators=150,
                        min_child_weight = 5,
                        reg_lambda=100,
                        use_label_encoder=False)
clf_xgb.fit(X_train, y_train)


plot_confusion_matrix(clf_xgb, X_test, y_test, display_labels=["Not Approved", "Approved"])

print("\n-----Out of sample test: XGBoost")


predicted = clf_xgb.predict(X_test) 
prob_approved = clf_xgb.predict_proba(X_test)
prob_approved = [x[1] for x in prob_approved] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, prob_approved))

