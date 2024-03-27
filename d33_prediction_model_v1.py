# Databricks notebook source
from numpy import mean
from numpy import std
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Load Data & Fill Null Values

# COMMAND ----------

# Load the data and convert to pandas
query_training_data = """
SELECT *
FROM churn_model_d33_training_data 
WHERE subscription_active_day IS TRUE
  AND days_since_first_sub_payment IN (7,14,21)
"""
df_training_data = spark.sql(query_training_data)
df_training_data_pd = df_training_data.toPandas()

# COMMAND ----------

# Show data info
df_training_data_pd.info()

# COMMAND ----------

# Select the columns to be used in the model
columns = ['STUDENT_AGE'
        , 'DAYS_SINCE_FIRST_SUB_PAYMENT'
        , 'N_DAYS_SINCE_LAST_CONFIRMED_LESSON'
        , 'NUM_CONFIRMED_LESSONS_DURING_SUB'
        , 'HOURS'
        , 'SUBJECT'
        , 'GMV_PROCEEDS_USD'
        , 'AVG_LESSON_RATING'
        , 'SUM_PAGES_VIEWED_ALL'
        , 'AVG_PAGES_VIEWED_ALL'
        , 'DAYS_WITH_SESSION'
        , 'SUM_SESSION_DURATION_ALL'
        , 'AVG_SESSION_DURATION_ALL'
        , 'SUM_PAGE_VIEWED_APP'
        , 'AVG_PAGE_VIEWED_APP'
        , 'SUM_SESSION_DURATION_APP'
        , 'AVG_SESSION_DURATION_APP'
        , 'SUM_PAGE_VIEWED_NON_APP'
        , 'AVG_PAGE_VIEWED_NON_APP'
        , 'SUM_SESSION_DURATION_NON_APP'
        , 'AVG_SESSION_DURATION_NON_APP'
        , 'DAYS_SINCE_LAST_SESSION'
        , 'STUDENT_MESSAGES_SENT'
        , 'TUTOR_MESSAGES_SENT'
        , 'COUNTRY_CODE'
        , 'FOCUS_COUNTRIES'
        , 'PREPLY_REGION'
        , 'LANGUAGE_VERSION'
        , 'NUM_SEARCH_PAGE_VIEWS'
        , 'NUM_IMPRESSIONS']
pred_columns = 'CHURN_DAY_33'

# COMMAND ----------

# Fill null values
df_training_data_non_null_pd = df_training_data_pd[columns + [pred_columns]]

numerical = ['STUDENT_AGE'
        , 'N_DAYS_SINCE_LAST_CONFIRMED_LESSON'
        , 'NUM_CONFIRMED_LESSONS_DURING_SUB'
        , 'HOURS'
        , 'GMV_PROCEEDS_USD'
        , 'AVG_LESSON_RATING'
        , 'SUM_PAGES_VIEWED_ALL'
        , 'AVG_PAGES_VIEWED_ALL'
        , 'DAYS_WITH_SESSION'
        , 'SUM_SESSION_DURATION_ALL'
        , 'AVG_SESSION_DURATION_ALL'
        , 'SUM_PAGE_VIEWED_APP'
        , 'AVG_PAGE_VIEWED_APP'
        , 'SUM_SESSION_DURATION_APP'
        , 'AVG_SESSION_DURATION_APP'
        , 'SUM_PAGE_VIEWED_NON_APP'
        , 'AVG_PAGE_VIEWED_NON_APP'
        , 'SUM_SESSION_DURATION_NON_APP'
        , 'AVG_SESSION_DURATION_NON_APP'
        , 'DAYS_SINCE_LAST_SESSION'
        , 'STUDENT_MESSAGES_SENT'
        , 'TUTOR_MESSAGES_SENT'
        , 'NUM_SEARCH_PAGE_VIEWS'
        , 'NUM_IMPRESSIONS'
]
df_training_data_non_null_pd[numerical] = df_training_data_non_null_pd[numerical].fillna(0).astype('float')

categorical = [ 'SUBJECT'
        , 'COUNTRY_CODE'
        , 'FOCUS_COUNTRIES'
        , 'PREPLY_REGION'
        , 'LANGUAGE_VERSION'
]
df_training_data_non_null_pd[categorical] = df_training_data_non_null_pd[categorical].fillna('empty').astype('category')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Correlation plot

# COMMAND ----------

# Correlation Plot
corr = df_training_data_non_null_pd[columns + [pred_columns]].corr()
corr.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Training
# MAGIC ## 3.1 Data for 7 days since subscription start

# COMMAND ----------

# Filter users on their 7th day since sub start
days_since_sub_payment = 7
df_training_data_non_null_pd_days_7 = df_training_data_non_null_pd[df_training_data_non_null_pd.DAYS_SINCE_FIRST_SUB_PAYMENT == days_since_sub_payment]

# Split into train and test sets
X_7 = df_training_data_non_null_pd_days_7[columns]
y_7 = df_training_data_non_null_pd_days_7[pred_columns]
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(X_7, y_7, test_size=0.33, random_state=123)

# Balance the dataset
df_train_7 = X_train_7
df_train_7['y_train'] = y_train_7
num_churn_7 = df_train_7[y_train_7 == True].shape[0]
non_churn_7 = df_train_7[y_train_7 == True]
churn_7 = df_train_7[y_train_7 == False].sample(num_churn_7)
df_train_weighted_7 = pd.concat([non_churn_7,churn_7])
X_train_w_7 = df_train_weighted_7[columns]
y_train_w_7 =  df_train_weighted_7['y_train'].values

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1.1 Logistic Regression, Decision Tree & Gradient Boosting Classifier (7 days)

# COMMAND ----------


# Prepare the training pipeline
one_hot_encoder = make_column_transformer(
    (
        OneHotEncoder( handle_unknown="ignore"),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

# Train logistic regression
train_pipeline_7_logistic_regression = make_pipeline(
    one_hot_encoder 
    ,  LogisticRegression(random_state=123,max_iter= 1000)
)
train_pipeline_7_logistic_regression.fit(X_train_w_7,y_train_w_7)

# Train decision tree
train_pipeline_7_decision_tree = make_pipeline(
    one_hot_encoder 
    ,  DecisionTreeClassifier(random_state=123)
)
train_pipeline_7_decision_tree.fit(X_train_w_7,y_train_w_7)

# Train gradient boosting
train_pipeline_7_gradient_boosting = make_pipeline(
    one_hot_encoder 
    ,  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=123)
)
train_pipeline_7_gradient_boosting.fit(X_train_w_7,y_train_w_7)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Data for 14 days since subscription start

# COMMAND ----------

# Filter users on their 14th day since sub start
days_since_sub_payment = 14
df_training_data_non_null_pd_days_14 = df_training_data_non_null_pd[df_training_data_non_null_pd.DAYS_SINCE_FIRST_SUB_PAYMENT == days_since_sub_payment]

# Split into train and test sets
X_14 = df_training_data_non_null_pd_days_14[columns]
y_14 = df_training_data_non_null_pd_days_14[pred_columns]
X_train_14, X_test_14, y_train_14, y_test_14 = train_test_split(X_14, y_14, test_size=0.33, random_state=123)

# Balance the dataset
df_train_14 = X_train_14
df_train_14['y_train'] = y_train_14
num_churn_14 = df_train_14[y_train_14 == True].shape[0]
non_churn_14 = df_train_14[y_train_14 == True]
churn_14 = df_train_14[y_train_14 == False].sample(num_churn_14)
df_train_weighted_14 = pd.concat([non_churn_14,churn_14])
X_train_w_14 = df_train_weighted_14[columns]
y_train_w_14 =  df_train_weighted_14['y_train'].values

# COMMAND ----------

# Prepare the training pipeline
one_hot_encoder = make_column_transformer(
    (
        OneHotEncoder( handle_unknown="ignore"),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

# Train logistic regression
train_pipeline_14_logistic_regression = make_pipeline(
    one_hot_encoder 
    ,  LogisticRegression(random_state=123,max_iter= 1000)
)
train_pipeline_14_logistic_regression.fit(X_train_w_14,y_train_w_14)

# Train decision tree
train_pipeline_14_decision_tree = make_pipeline(
    one_hot_encoder 
    ,  DecisionTreeClassifier(random_state=123)
)
train_pipeline_14_decision_tree.fit(X_train_w_14,y_train_w_14)

# Train gradient boosting
train_pipeline_14_gradient_boosting = make_pipeline(
    one_hot_encoder 
    ,  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=123)
)
train_pipeline_14_gradient_boosting.fit(X_train_w_14,y_train_w_14)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Data for 21 days since subscription start

# COMMAND ----------

# Filter users on their 21th day since sub start
days_since_sub_payment = 21
df_training_data_non_null_pd_days_21 = df_training_data_non_null_pd[df_training_data_non_null_pd.DAYS_SINCE_FIRST_SUB_PAYMENT == days_since_sub_payment]

# Split into train and test sets
X_21 = df_training_data_non_null_pd_days_21[columns]
y_21 = df_training_data_non_null_pd_days_21[pred_columns]
X_train_21, X_test_21, y_train_21, y_test_21 = train_test_split(X_21, y_21, test_size=0.33, random_state=123)

# Balance the dataset
df_train_21 = X_train_21
df_train_21['y_train'] = y_train_21
num_churn_21 = df_train_21[y_train_21 == True].shape[0]
non_churn_21 = df_train_21[y_train_21 == True]
churn_21 = df_train_21[y_train_21 == False].sample(num_churn_21)
df_train_weighted_21 = pd.concat([non_churn_21,churn_21])
X_train_w_21 = df_train_weighted_21[columns]
y_train_w_21 =  df_train_weighted_21['y_train'].values

# COMMAND ----------


# Prepare the training pipeline
one_hot_encoder = make_column_transformer(
    (
        OneHotEncoder( handle_unknown="ignore"),
        make_column_selector(dtype_include="category"),
    ),
    remainder="passthrough",
)

# Train logistic regression
train_pipeline_21_logistic_regression = make_pipeline(
    one_hot_encoder 
    ,  LogisticRegression(random_state=123,max_iter= 2000)
)
train_pipeline_21_logistic_regression.fit(X_train_w_21,y_train_w_21)

# Train decision tree
train_pipeline_21_decision_tree = make_pipeline(
    one_hot_encoder 
    ,  DecisionTreeClassifier(random_state=123)
)
train_pipeline_21_decision_tree.fit(X_train_w_21,y_train_w_21)

# Train gradient boosting
train_pipeline_21_gradient_boosting = make_pipeline(
    one_hot_encoder 
    ,  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=123)
)
train_pipeline_21_gradient_boosting.fit(X_train_w_21,y_train_w_21)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Model Validation

# COMMAND ----------

print('Precision_score 7 days Log Reg: ',precision_score(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7))
print('Precision_score 7 days decision_tree: ',precision_score(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7))
print('Precision_score 7 days Grad Boost: ',precision_score(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7))
print('Precision_score 14 days Log Reg: ',precision_score(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14))
print('Precision_score 14 days decision_tree: ',precision_score(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14))
print('Precision_score 14 days Grad Boost: ',precision_score(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14))
print('Precision_score 21 days Log Reg: ',precision_score(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21))
print('Precision_score 21 days decision_tree: ',precision_score(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21))
print('Precision_score 21 days Grad Boost: ',precision_score(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21))
precision_values = ['precision_score'
    ,precision_score(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7)
    ,precision_score(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7)
    ,precision_score(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7)
    ,precision_score(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14)
    ,precision_score(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14)
    ,precision_score(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14)
    ,precision_score(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21)
    ,precision_score(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21)
    ,precision_score(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21)
]

# COMMAND ----------

print('recall_score 7 days Log Reg: ',recall_score(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7))
print('recall_score 7 days decision_tree: ',recall_score(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7))
print('recall_score 7 days Grad Boost: ',recall_score(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7))
print('recall_score 14 days Log Reg: ',recall_score(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14))
print('recall_score 14 days decision_tree: ',recall_score(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14))
print('recall_score 14 days Grad Boost: ',recall_score(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14))
print('recall_score 21 days Log Reg: ',recall_score(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21))
print('recall_score 21 days decision_tree: ',recall_score(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21))
print('recall_score 21 days Grad Boost: ',recall_score(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21))
recall_values = ['recall_score'
    ,recall_score(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7)
    ,recall_score(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7)
    ,recall_score(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7)
    ,recall_score(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14)
    ,recall_score(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14)
    ,recall_score(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14)
    ,recall_score(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21)
    ,recall_score(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21)
    ,recall_score(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21)
]

# COMMAND ----------

print('roc_auc_score 7 days Log Reg: ',roc_auc_score(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7))
print('roc_auc_score 7 days decision_tree: ',roc_auc_score(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7))
print('roc_auc_score 7 days Grad Boost: ',roc_auc_score(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7))
print('roc_auc_score 14 days Log Reg: ',roc_auc_score(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14))
print('roc_auc_score 14 days decision_tree: ',roc_auc_score(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14))
print('roc_auc_score 14 days Grad Boost: ',roc_auc_score(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14))
print('roc_auc_score 21 days Log Reg: ',roc_auc_score(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21))
print('roc_auc_score 21 days decision_tree: ',roc_auc_score(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21))
print('roc_auc_score 21 days Grad Boost: ',roc_auc_score(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21))
roc_auc_score_values = [ 'roc_auc_score'
    ,roc_auc_score(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7)
    ,roc_auc_score(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7)
    ,roc_auc_score(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7)
    ,roc_auc_score(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14)
    ,roc_auc_score(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14)
    ,roc_auc_score(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14)
    ,roc_auc_score(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21)
    ,roc_auc_score(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21)
    ,roc_auc_score(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21)
]

# COMMAND ----------

# tn, fp, fn, tp
print('tn, fp, fn, tp')
print('confusion_matrix 7 days Log Reg: ',confusion_matrix(train_pipeline_7_logistic_regression.predict(X_test_7),y_test_7))
print('confusion_matrix 7 days decision_tree: ',confusion_matrix(train_pipeline_7_decision_tree.predict(X_test_7),y_test_7))
print('confusion_matrix 7 days Grad Boost: ',confusion_matrix(train_pipeline_7_gradient_boosting.predict(X_test_7),y_test_7))
print('confusion_matrix 14 days Log Reg: ',confusion_matrix(train_pipeline_14_logistic_regression.predict(X_test_14),y_test_14))
print('confusion_matrix 14 days decision_tree: ',confusion_matrix(train_pipeline_14_decision_tree.predict(X_test_14),y_test_14))
print('confusion_matrix 14 days Grad Boost: ',confusion_matrix(train_pipeline_14_gradient_boosting.predict(X_test_14),y_test_14))
print('confusion_matrix 21 days Log Reg: ',confusion_matrix(train_pipeline_21_logistic_regression.predict(X_test_21),y_test_21))
print('confusion_matrix 21 days decision_tree: ',confusion_matrix(train_pipeline_21_decision_tree.predict(X_test_21),y_test_21))
print('confusion_matrix 21 days Grad Boost: ',confusion_matrix(train_pipeline_21_gradient_boosting.predict(X_test_21),y_test_21))

# COMMAND ----------

columns = ['metric','linear_regression_7','decision_tree_7','gradient_boosting_7','linear_regression_14','decision_tree_14','gradient_boosting_14','linear_regression_21','decision_tree_21','gradient_boosting_21']
values = [precision_values,recall_values,roc_auc_score_values]

df_metrics = pd.DataFrame( columns = columns, data = values)

# COMMAND ----------

display(df_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Threshold validation: Gradient Boosting

# COMMAND ----------

y_test_pred_prob_7_grad_boost_true = train_pipeline_7_gradient_boosting.predict_proba(X_test_7)[:,1]
y_test_pred_prob_14_grad_boost_true = train_pipeline_14_gradient_boosting.predict_proba(X_test_14)[:,1]
y_test_pred_prob_21_grad_boost_true = train_pipeline_21_gradient_boosting.predict_proba(X_test_21)[:,1]

# COMMAND ----------

df_y_pred_7 = pd.DataFrame()
df_y_pred_14 = pd.DataFrame()
df_y_pred_21 = pd.DataFrame()

df_y_pred_7['y_test_pred_prob_7_grad_boost_true'] = y_test_pred_prob_7_grad_boost_true
df_y_pred_14['y_test_pred_prob_14_grad_boost_true'] = y_test_pred_prob_14_grad_boost_true
df_y_pred_21['y_test_pred_prob_21_grad_boost_true'] = y_test_pred_prob_21_grad_boost_true

df_y_pred_7['y_test'] = y_test_7
df_y_pred_14['y_test'] = y_test_14
df_y_pred_21['y_test'] = y_test_21

df_y_pred_7['90_thr_7'] = df_y_pred_7.apply(lambda x: True if x['y_test_pred_prob_7_grad_boost_true'] >=0.9 else False, axis = 1)
df_y_pred_7['85_thr_7'] = df_y_pred_7.apply(lambda x: True if x['y_test_pred_prob_7_grad_boost_true'] >=0.85 else False, axis = 1)
df_y_pred_7['80_thr_7'] = df_y_pred_7.apply(lambda x: True if x['y_test_pred_prob_7_grad_boost_true'] >=0.8 else False, axis = 1)
df_y_pred_7['75_thr_7'] = df_y_pred_7.apply(lambda x: True if x['y_test_pred_prob_7_grad_boost_true'] >=0.75 else False, axis = 1)
df_y_pred_7['70_thr_7'] = df_y_pred_7.apply(lambda x: True if x['y_test_pred_prob_7_grad_boost_true'] >=0.7 else False, axis = 1)
df_y_pred_7['60_thr_7'] = df_y_pred_7.apply(lambda x: True if x['y_test_pred_prob_7_grad_boost_true'] >=0.6 else False, axis = 1)

df_y_pred_14['90_thr_14'] = df_y_pred_14.apply(lambda x: True if x['y_test_pred_prob_14_grad_boost_true'] >=0.9 else False, axis = 1)
df_y_pred_14['85_thr_14'] = df_y_pred_14.apply(lambda x: True if x['y_test_pred_prob_14_grad_boost_true'] >=0.85 else False, axis = 1)
df_y_pred_14['80_thr_14'] = df_y_pred_14.apply(lambda x: True if x['y_test_pred_prob_14_grad_boost_true'] >=0.8 else False, axis = 1)
df_y_pred_14['75_thr_14'] = df_y_pred_14.apply(lambda x: True if x['y_test_pred_prob_14_grad_boost_true'] >=0.75 else False, axis = 1)
df_y_pred_14['70_thr_14'] = df_y_pred_14.apply(lambda x: True if x['y_test_pred_prob_14_grad_boost_true'] >=0.7 else False, axis = 1)
df_y_pred_14['60_thr_14'] = df_y_pred_14.apply(lambda x: True if x['y_test_pred_prob_14_grad_boost_true'] >=0.6 else False, axis = 1)

df_y_pred_21['90_thr_21'] = df_y_pred_21.apply(lambda x: True if x['y_test_pred_prob_21_grad_boost_true'] >=0.9 else False, axis = 1)
df_y_pred_21['85_thr_21'] = df_y_pred_21.apply(lambda x: True if x['y_test_pred_prob_21_grad_boost_true'] >=0.85 else False, axis = 1)
df_y_pred_21['80_thr_21'] = df_y_pred_21.apply(lambda x: True if x['y_test_pred_prob_21_grad_boost_true'] >=0.8 else False, axis = 1)
df_y_pred_21['75_thr_21'] = df_y_pred_21.apply(lambda x: True if x['y_test_pred_prob_21_grad_boost_true'] >=0.75 else False, axis = 1)
df_y_pred_21['70_thr_21'] = df_y_pred_21.apply(lambda x: True if x['y_test_pred_prob_21_grad_boost_true'] >=0.7 else False, axis = 1)
df_y_pred_21['60_thr_21'] = df_y_pred_21.apply(lambda x: True if x['y_test_pred_prob_21_grad_boost_true'] >=0.6 else False, axis = 1)

# COMMAND ----------

print('Recall at 90% thr 7 days: ', recall_score(y_test_7, df_y_pred_7['90_thr_7']))
print('Recall at 85% thr 7 days: ', recall_score(y_test_7, df_y_pred_7['85_thr_7']))
print('Recall at 80% thr 7 days: ', recall_score(y_test_7, df_y_pred_7['80_thr_7']))
print('Recall at 75% thr 7 days: ', recall_score(y_test_7, df_y_pred_7['75_thr_7']))
print('Recall at 70% thr 7 days: ', recall_score(y_test_7, df_y_pred_7['70_thr_7']))
print('Recall at 60% thr 7 days: ', recall_score(y_test_7, df_y_pred_7['60_thr_7']))

print('Recall at 90% thr 14 days: ', recall_score(y_test_14, df_y_pred_14['90_thr_14']))
print('Recall at 85% thr 14 days: ', recall_score(y_test_14, df_y_pred_14['85_thr_14']))
print('Recall at 80% thr 14 days: ', recall_score(y_test_14, df_y_pred_14['80_thr_14']))
print('Recall at 75% thr 14 days: ', recall_score(y_test_14, df_y_pred_14['75_thr_14']))
print('Recall at 70% thr 14 days: ', recall_score(y_test_14, df_y_pred_14['70_thr_14']))
print('Recall at 60% thr 14 days: ', recall_score(y_test_14, df_y_pred_14['60_thr_14']))

print('Recall at 90% thr 21 days: ', recall_score(y_test_21, df_y_pred_21['90_thr_21']))
print('Recall at 85% thr 21 days: ', recall_score(y_test_21, df_y_pred_21['85_thr_21']))
print('Recall at 80% thr 21 days: ', recall_score(y_test_21, df_y_pred_21['80_thr_21']))
print('Recall at 75% thr 21 days: ', recall_score(y_test_21, df_y_pred_21['75_thr_21']))
print('Recall at 70% thr 21 days: ', recall_score(y_test_21, df_y_pred_21['70_thr_21']))
print('Recall at 60% thr 21 days: ', recall_score(y_test_21, df_y_pred_21['60_thr_21']))

recall_different_thr = ['Recall'
, recall_score(y_test_7, df_y_pred_7['90_thr_7'])
, recall_score(y_test_7, df_y_pred_7['85_thr_7'])
, recall_score(y_test_7, df_y_pred_7['80_thr_7'])
, recall_score(y_test_7, df_y_pred_7['75_thr_7'])
, recall_score(y_test_7, df_y_pred_7['70_thr_7'])
, recall_score(y_test_7, df_y_pred_7['60_thr_7'])
, recall_score(y_test_14, df_y_pred_14['90_thr_14'])
, recall_score(y_test_14, df_y_pred_14['85_thr_14'])
, recall_score(y_test_14, df_y_pred_14['80_thr_14'])
, recall_score(y_test_14, df_y_pred_14['75_thr_14'])
, recall_score(y_test_14, df_y_pred_14['70_thr_14'])
, recall_score(y_test_14, df_y_pred_14['60_thr_14'])
, recall_score(y_test_21, df_y_pred_21['90_thr_21'])
, recall_score(y_test_21, df_y_pred_21['85_thr_21'])
, recall_score(y_test_21, df_y_pred_21['80_thr_21'])
, recall_score(y_test_21, df_y_pred_21['75_thr_21'])
, recall_score(y_test_21, df_y_pred_21['70_thr_21'])
, recall_score(y_test_21, df_y_pred_21['60_thr_21'])
]

# COMMAND ----------

print('Precision at 90% thr: ', precision_score(y_test_7, df_y_pred_7['90_thr_7']))
print('Precision at 85% thr: ', precision_score(y_test_7, df_y_pred_7['85_thr_7']))
print('Precision at 80% thr: ', precision_score(y_test_7, df_y_pred_7['80_thr_7']))
print('Precision at 75% thr: ', precision_score(y_test_7, df_y_pred_7['75_thr_7']))
print('Precision at 70% thr: ', precision_score(y_test_7, df_y_pred_7['70_thr_7']))
print('Precision at 60% thr: ', precision_score(y_test_7, df_y_pred_7['60_thr_7']))

print('Precision at 90% thr 14 days: ', precision_score(y_test_14, df_y_pred_14['90_thr_14']))
print('Precision at 85% thr 14 days: ', precision_score(y_test_14, df_y_pred_14['85_thr_14']))
print('Precision at 80% thr 14 days: ', precision_score(y_test_14, df_y_pred_14['80_thr_14']))
print('Precision at 75% thr 14 days: ', precision_score(y_test_14, df_y_pred_14['75_thr_14']))
print('Precision at 70% thr 14 days: ', precision_score(y_test_14, df_y_pred_14['70_thr_14']))
print('Precision at 60% thr 14 days: ', precision_score(y_test_14, df_y_pred_14['60_thr_14']))

print('Precision at 90% thr 21 days: ', precision_score(y_test_21, df_y_pred_21['90_thr_21']))
print('Precision at 85% thr 21 days: ', precision_score(y_test_21, df_y_pred_21['85_thr_21']))
print('Precision at 80% thr 21 days: ', precision_score(y_test_21, df_y_pred_21['80_thr_21']))
print('Precision at 75% thr 21 days: ', precision_score(y_test_21, df_y_pred_21['75_thr_21']))
print('Precision at 70% thr 21 days: ', precision_score(y_test_21, df_y_pred_21['70_thr_21']))
print('Precision at 60% thr 21 days: ', precision_score(y_test_21, df_y_pred_21['60_thr_21']))

precision_different_thr = [ 'Precision'
, precision_score(y_test_7, df_y_pred_7['90_thr_7'])
, precision_score(y_test_7, df_y_pred_7['85_thr_7'])
, precision_score(y_test_7, df_y_pred_7['80_thr_7'])
, precision_score(y_test_7, df_y_pred_7['75_thr_7'])
, precision_score(y_test_7, df_y_pred_7['70_thr_7'])
, precision_score(y_test_7, df_y_pred_7['60_thr_7'])
, precision_score(y_test_14, df_y_pred_14['90_thr_14'])
, precision_score(y_test_14, df_y_pred_14['85_thr_14'])
, precision_score(y_test_14, df_y_pred_14['80_thr_14'])
, precision_score(y_test_14, df_y_pred_14['75_thr_14'])
, precision_score(y_test_14, df_y_pred_14['70_thr_14'])
, precision_score(y_test_14, df_y_pred_14['60_thr_14'])
, precision_score(y_test_21, df_y_pred_21['90_thr_21'])
, precision_score(y_test_21, df_y_pred_21['85_thr_21'])
, precision_score(y_test_21, df_y_pred_21['80_thr_21'])
, precision_score(y_test_21, df_y_pred_21['75_thr_21'])
, precision_score(y_test_21, df_y_pred_21['70_thr_21'])
, precision_score(y_test_21, df_y_pred_21['60_thr_21'])
]

# COMMAND ----------

columns = ['metric','90_thr_7','85_thr_7','80_thr_7','75_thr_7','70_thr_7','60_thr_7','90_thr_14','85_thr_14','80_thr_14','75_thr_14','70_thr_14','60_thr_14','90_thr_21','85_thr_21','80_thr_21','75_thr_21','70_thr_21','60_thr_21']
df_thr_metrics = pd.DataFrame(columns = columns, data = [recall_different_thr,precision_different_thr])

# COMMAND ----------

display(df_thr_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Feature Importance

# COMMAND ----------


feature_importance_7 = pd.DataFrame()
feature_importance_7['feature_name'] = train_pipeline_7_gradient_boosting.steps[0][1].get_feature_names()
feature_importance_7['feature_importances'] = train_pipeline_7_gradient_boosting.steps[1][1].feature_importances_

feature_importance_14 = pd.DataFrame()
feature_importance_14['feature_name'] = train_pipeline_14_gradient_boosting.steps[0][1].get_feature_names()
feature_importance_14['feature_importances'] = train_pipeline_14_gradient_boosting.steps[1][1].feature_importances_

feature_importance_21 = pd.DataFrame()
feature_importance_21['feature_name'] = train_pipeline_21_gradient_boosting.steps[0][1].get_feature_names()
feature_importance_21['feature_importances'] = train_pipeline_21_gradient_boosting.steps[1][1].feature_importances_

# COMMAND ----------

display(feature_importance_7.sort_values(by = 'feature_importances',ascending = False))

# COMMAND ----------

display(feature_importance_14.sort_values(by = 'feature_importances',ascending = False))

# COMMAND ----------

display(feature_importance_21.sort_values(by = 'feature_importances',ascending = False))
