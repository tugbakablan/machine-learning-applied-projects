import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("6-bank_customers.csv")
print(data.head())
print(data.info())
print(data["subscribed"].unique())
print(data["subscribed"].value_counts())

X = data.drop("subscribed", axis=1)
y = data["subscribed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
ac = accuracy_score(y_test, y_pred)
print("LogisticReg Accuracy: ", ac)

penalty = ['l1', 'l2', 'elasticnet']
c_values = [100, 10, 1, 0.1, 0.01]
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
params = dict(penalty = penalty, C = c_values, solver = solver )

grid = GridSearchCV(estimator = logistic, param_grid = params, cv = 5, scoring = 'accuracy', n_jobs = -1)
grid.fit(X_train, y_train)
y_pred1 = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
ac1 = accuracy_score(y_test, y_pred1)
print("Grid Search Accuracy: " , ac1)

print(grid.best_params_)

rd = RandomizedSearchCV(estimator = logistic, param_distributions = params, n_jobs = -1, cv = 5, scoring = 'accuracy', n_iter = 200)
rd.fit(X_train, y_train)
y_pred2 = rd.predict(X_test)
print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
ac2 = accuracy_score(y_test, y_pred2)
print("Random Search Accuracy: " , ac2)

acs = [ac, ac1,ac2]
print(acs)