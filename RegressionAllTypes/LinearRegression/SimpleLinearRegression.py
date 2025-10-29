import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("1-studyhours.csv")
print(data.head())

plt.scatter(data['Study Hours'], data['Exam Score'], color='red')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')

X=data[['Study Hours']]
y=data['Exam Score']
print(type(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.info())
print(X_test.info())
print(y_train.info())
print(y_test.info())

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns=['Study Hours'])
print(X_train_df.head())
# or instead
print(X_train[:5, :])

linearreg = LinearRegression()
linearreg.fit(X_train, y_train)

print("Coefficients: ", linearreg.coef_)
print("Intercept: ", linearreg.intercept_)

fig = plt.figure()
plt.scatter(X_train,y_train)
plt.plot(X_train, linearreg.predict(X_train), c="red")

print(linearreg.predict(scaler.transform([[20]])))
print(linearreg.predict(scaler.transform([[10]])))
print(linearreg.predict(scaler.transform([[0]])))
print(linearreg.predict(scaler.transform(X_test)))

y_pred_test = linearreg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)
print("mse: " ,mse)
print("mae: ", mae)
print("rmse :", rmse )
print("r2: ", r2)

fig= plt.figure()
plt.scatter(y_test, y_pred_test)

print(1-((1-r2)*(len(y_test)-1)/len(y_test)-X_test.shape[1]-1)) #adjusted r2 example
plt.show()