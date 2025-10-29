import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("2-multiplegradesdataset.csv")
print(data.info())
print(data.corr)

sns.pairplot(data)
fig = plt.figure()
sns.regplot(x=data["Study Hours"], y=data["Exam Score"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#X = data[["Study Hours"],["Sleep Hours"],["Attendance Rate"],["Social Media Hours"]]
#y = data["Exam Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

regression = LinearRegression()
regression.fit(X_train, y_train)

print(data.iloc[0])
newData = [[5, 6, 70, 1]]
newDataScaled = scaler.transform(newData)

print(regression.predict(newDataScaled))

y_prediction = regression.predict(X_test)
mse = mean_squared_error(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_prediction)
adjusted_r2 = 1 - (((1-r2)*(len(y_test)-1))/len(y_test)-X_test.shape[1]-1)
print(mse)
print(mae)
print(rmse)
print(r2)
print(adjusted_r2)
print(regression.coef_)
print(regression.intercept_)

fig = plt.figure()
sns.regplot(x=y_test, y=y_prediction)

students= [
    [5, 6, 70, 1],
    [3, 9, 40, 0],
    [10, 3, 80, 2],
]
print(regression.predict(scaler.transform(students)))

plt.show()