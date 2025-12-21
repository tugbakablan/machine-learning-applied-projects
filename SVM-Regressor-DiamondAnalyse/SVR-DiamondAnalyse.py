import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('diamonds.csv')
print(data.head())
print(data.info())

data = data.drop("Unnamed: 0", axis = 1)
print(data.isnull().sum())
print(data.describe())

print(len(data[data["x"] == 0]), len(data[data["y"] == 0]), len(data[data["z"] == 0]))
data = data.drop(data[data["x"] == 0].index)
data = data.drop(data[data["y"] == 0].index)
data = data.drop(data[data["z"] == 0].index)
print(data.describe())

sns.pairplot(data)
sns.scatterplot(y= data["price"], x = data["y"])
sns.scatterplot(y= data["price"], x = data["z"])
sns.scatterplot(y= data["price"], x = data["depth"])
sns.scatterplot(y= data["price"], x = data["table"])

data = data[data["y"] <= 20]
data = data[(data["z"] > 2) & (data["z"] < 15)]
data = data[(data["table"] < 75) & (data["table"] > 25)]
data = data[(data["depth"] < 75) & (data["depth"] > 50)]
print(data.info())
print(data.describe())

print(data["cut"].value_counts(),
      data["color"].value_counts(),
      data["clarity"].value_counts())

X = data.drop(["price"], axis = 1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 12, test_size = 0.20)

label_encoder = LabelEncoder()
for col in ["cut", "color", "clarity"]:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])
print(X_train.head())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred_1 = linear.predict(X_test_scaled)

mae=mean_absolute_error(y_test,y_pred_1)
score=r2_score(y_test,y_pred_1)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred_1)

svr = SVR()
svr.fit(X_train_scaled, y_train)
y_pred_2 = svr.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred_2)
score=r2_score(y_test,y_pred_2)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred_2)

param_grid = {
    'C' : [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf','linear']
}
grid = GridSearchCV(SVR(), param_grid, verbose = 2, n_jobs = -1, refit = True)
# grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)

y_pred=grid.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred)
plt.show()




