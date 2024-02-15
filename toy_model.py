import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
#from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/jovyan/hfactory_magic_folders/financial_graph_mining_for_customers___supply_chains_assessment/static_data_all_x.csv',delimiter = ";",header = 0)

#print(df.head(3))

df4 = df[df['QUARTER'] == 'q4'][['ID' ,'REGION',  'ESG_SCORE', 'CATEGORY' ,'T_LOCAL_MT_ACTIF_SOCIAL', 'T_LOCAL_TX_PD']]
df3 = df[df['QUARTER'] == 'q3'][['ID', 'T_LOCAL_TX_PD']]

merged_df = pd.merge(df3, df4, on='ID', how='inner')

merged_df.rename(columns={'T_LOCAL_TX_PD_x': 'T_LOCAL_TX_PD_q3', 'T_LOCAL_TX_PD_y': 'T_LOCAL_TX_PD_q4'}, inplace=True)

count_equal = (merged_df['T_LOCAL_TX_PD_q3'] == merged_df['T_LOCAL_TX_PD_q4']).sum()

print('Default accuracy : ',count_equal/len(merged_df))

#print(merged_df.head(3))

print(merged_df.head(10))

X = merged_df.iloc[:, 1:6]  # 1 to 5 columns as features
y = merged_df.iloc[:, 6]   # 6th column as target


#print(type(y))
#print(type(X.iloc[:,0]))
#print(X.iloc[:,0].head())
label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([y,X.iloc[:,0]]))
y = label_encoder.transform(y)

num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)

#print(X.head(3))
#print(y[0:3])

columns_to_encode = [0, 1, 3]

# Extract the columns to be one-hot encoded
columns_subset = X.iloc[:, columns_to_encode]

# Perform one-hot encoding on the subset of columns
encoded_columns = pd.get_dummies(columns_subset, drop_first=True)  # Set drop_first=True to avoid multicollinearity

# Drop the original columns from the DataFrame
df = X.drop(columns_subset.columns, axis=1)

# Concatenate the original DataFrame with the encoded columns
df = pd.concat([df, encoded_columns], axis=1)

from sklearn.preprocessing import StandardScaler

# Assuming 'df' is your DataFrame
# Select numerical columns to scale
numerical_cols = df.select_dtypes(include=['int', 'float']).columns

# Create a copy of the DataFrame to avoid modifying the original DataFrame
df_scaled = df.copy()

# Create a StandardScaler object
scaler = StandardScaler()

#print(df.head(3))

df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#print(df_scaled.head(3))

X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#rf_classifier = GradientBoostingClassifier()

rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)

# Fit the classifier on the training set
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using all variables:", accuracy)

print("Balanced accuracy",balanced_accuracy_score(y_test, y_pred))


rf_classifier = RandomForestClassifier(random_state=42)
#rf_classifier = LogisticRegression()

#print(X.iloc[:,0].head(3))
new_X = pd.get_dummies(X.iloc[:,0])

#print(new_X.head(3))

print(len(new_X.columns))

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42)

# Fit the classifier on the training set
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


print("Accuracy using only PD_q3 and model:", accuracy)

print("Balanced accuracy",balanced_accuracy_score(y_test, y_pred))

y_pred = label_encoder.transform(X.iloc[:,0])

print("NOW USING DIRECTLY PD_q3 (WHITHOUT BUILDING A MODEL)")

accuracy = accuracy_score(y, y_pred)
print("Accuracy using directly PD_q3:", accuracy)

print("Balanced accuracy",balanced_accuracy_score(y, y_pred))