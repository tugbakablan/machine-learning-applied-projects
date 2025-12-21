import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC

data = pd.read_csv('Iris.csv')
print(data.head())
print(data["Species"].value_counts())

sns.pairplot(data=data, hue="Species")
sns.pairplot(data=data, hue="Species", diag_kind="hist")
sns.pairplot(data=data)

print(data.columns)
sns.scatterplot(x=data["SepalLengthCm"], y=data["SepalWidthCm"], hue=data["Species"])
sns.scatterplot(x=data["PetalLengthCm"], y=data["PetalWidthCm"], hue=data["Species"])
plt.show()

data = data.drop(["Id"], axis=1)
print(data.head())
label = LabelEncoder()
data["Species"] = label.fit_transform(data["Species"])
print(data["Species"].value_counts())

X = data.drop(["Species"], axis=1)
y = data["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred = gnb.predict(X_test_scaled)
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))
print("accuracy score: ", accuracy_score(y_pred, y_test))
print("classification report: ", classification_report(y_pred, y_test))

svc = SVC()
svc.fit(X_train_scaled, y_train)
y_pred_1 = svc.predict(X_test_scaled)
print("confusion matrix: \n", confusion_matrix(y_pred_1, y_test))
print("accuracy score: ", accuracy_score(y_pred_1, y_test))
print("classification report: ", classification_report(y_pred_1, y_test))

gnb_new = GaussianNB()
gnb_new.fit(X_train, y_train)
y_pred_2 = gnb_new.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_pred_2, y_test))
print("accuracy score: ", accuracy_score(y_pred_2, y_test))
print("classification report: ", classification_report(y_pred_2, y_test))

