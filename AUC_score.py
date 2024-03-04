import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/jovyan/hfactory_magic_folders/financial_graph_mining_for_customers___supply_chains_assessment/static_data_all_x.csv',delimiter = ";",header = 0)

#print(df.head(3))

df4 = df[df['QUARTER'] == 'q4'][['ID' ,'T_LOCAL_MT_ACTIF_SOCIAL', 'T_LOCAL_TX_PD']]
df3 = df[df['QUARTER'] == 'q3'][['ID' ,'REGION',  'ESG_SCORE', 'CATEGORY'  ,'T_LOCAL_MT_ACTIF_SOCIAL', 'T_LOCAL_TX_PD']]


merged_df = pd.merge(df3, df4, on='ID', how='inner')

print(merged_df)

def remove_sign(s):
    if s[0]=="-":
        return int(s[1:]),-1
    if s[-1]=='+':
        return int(s[:-1]),+1
    return int(s),0

def remove_sign_2(s):
    if s[0]=="-":
        return int(s[1:])-0.25
    if s[-1]=='+':
        return int(s[:-1])+0.25
    return int(s)

def build(x,y):
    if x==y:
        return 0
    a,b = remove_sign(x)
    c,d = remove_sign(y)
    if a==c:
        if b=='-':
            return 1
        else:
            return 1
    if a<c:
        return 1
    else:
        return 1

y = merged_df.apply(lambda row: build(row['T_LOCAL_TX_PD_x'], row['T_LOCAL_TX_PD_y']), axis=1)


frequency = y.value_counts()

print(frequency)

print(y)

X = merged_df.iloc[:,1:7]

label_encoder_X = LabelEncoder()
X.iloc[:, 0] = label_encoder_X.fit_transform(X.iloc[:, 0])

num_classes = len(label_encoder_X.classes_)
print("Number of REGIONS:", num_classes)

label_encoder_X = LabelEncoder()
X.iloc[:, 2] = label_encoder_X.fit_transform(X.iloc[:, 2])

num_classes = len(label_encoder_X.classes_)
print("Number of CATEGORIES:", num_classes)

columns_to_encode = [0, 2]

# Extract the columns to be one-hot encoded
columns_subset = X.iloc[:, columns_to_encode]

# Perform one-hot encoding on the subset of columns
encoded_columns = pd.get_dummies(columns_subset, drop_first=True)  # Set drop_first=True to avoid multicollinearity

# Drop the original columns from the DataFrame
df = X.drop(columns_subset.columns, axis=1)

# Concatenate the original DataFrame with the encoded columns
df = pd.concat([df, encoded_columns], axis=1)

df['T_LOCAL_TX_PD_x'] = df['T_LOCAL_TX_PD_x'].apply(remove_sign_2)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print(merged_df)

class_weights = {0: 1, 1: 20}

#print(class_weights)

#from sklearn.naive_bayes import GaussianNB

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=2,class_weight=class_weights)
#rf_classifier = LogisticRegression(class_weight=class_weights)


print(rf_classifier.class_weight)

# Fit the classifier on the training set
rf_classifier.fit(X_train, y_train)

y_tot = rf_classifier.predict(df)
# Calculate accuracy
accuracy = accuracy_score(y, y_tot)
print("Accuracy using all variables:", accuracy)

print("Balanced accuracy",balanced_accuracy_score(y, y_tot))

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

print(sum(y_test))
print(y_pred[0:5])
print(y_test[0:5])
print(sum(y_pred))

print('ACCURACY ON TEST SET : ')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using all variables:", accuracy)

print("Balanced accuracy",balanced_accuracy_score(y_test, y_pred))

print(rf_classifier.predict_proba(df))

import sys

# Your script code here

# When you want to exit the script
#sys.exit()

pred_proba = rf_classifier.predict_proba(X_test)

pred_proba = pred_proba[:, 1]

print(pred_proba[0:5])

constant = [0]*len(X_test)

from sklearn.metrics import roc_auc_score

# Assuming y_true contains the true labels and y_pred contains the predicted probabilities for positive class
auc = roc_auc_score(y_test, pred_proba)

print("AUC:", auc)

auc = roc_auc_score(y_test, constant)

print("AUC:", auc)