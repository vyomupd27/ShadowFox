import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:\\Users\\vyomu\\OneDrive\\Desktop\\ShadowFox\\TASK 2 - Intermediate\\loan_prediction.csv")
data.head(5)
data.info()
print(data.isnull().sum())

data = data.drop(["Loan_ID"], axis=1)
data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)

data["Dependents"] = (
    data["Dependents"]
    .replace("3+", "3")
    .fillna(data["Dependents"].mode()[0])  
    .astype(int)
)

data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0], inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0], inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0], inplace=True)

data["Gender"] = data["Gender"].map({"Male": 1, "Female" : 0})
data["Married"] = data["Married"].map({"Yes": 1, "No" : 0})
data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate" : 0})
data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No" : 0})
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N" : 0})

le = LabelEncoder()
data["Property_Area"] = le.fit_transform(data["Property_Area"])

X = data[["Credit_History","Dependents","Education","Gender","Loan_Amount_Term","LoanAmount","Married","Property_Area","Self_Employed","ApplicantIncome","CoapplicantIncome"]]
Y = data["Loan_Status"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

scaler = StandardScaler()

X_trained_scaled = scaler.fit_transform(X_train)
X_tested_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_trained_scaled, Y_train)

prediction = model.predict(X_tested_scaled)
prob       = model.predict_proba(X_tested_scaled)[:, 1]

print("Accuracy:", accuracy_score(Y_test, prediction))
print("Precision:", precision_score(Y_test, prediction))
print("Recall:", recall_score(Y_test, prediction))
print("F1 Score:", f1_score(Y_test, prediction))
print("Confusion Matrix:\n", confusion_matrix(Y_test, prediction))
print("ROC-AUC:  ", roc_auc_score(Y_test, prob))