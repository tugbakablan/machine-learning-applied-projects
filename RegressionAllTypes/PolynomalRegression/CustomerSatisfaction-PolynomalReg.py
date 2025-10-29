import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("3-customersatisfaction.csv")
print(data.info())
data.drop("Unnamed: 0",axis=1, inplace=True)
print(data.head())

plt.scatter(data["Customer Satisfaction"], data["Incentive"], color="green")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")

fig = plt.figure()
sns.regplot(x=data["Customer Satisfaction"], y=data["Incentive"], color="red")

X = data[["Customer Satisfaction"]]
y =data["Incentive"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressor = LinearRegression() #the compare linar and polynomal reg success results
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
adjusted_r2 = 1 - (((1-r2)*(len(y_test) - 1))/ len(y_test)- X_test.shape[1] - 1)
coef = regressor.coef_
intercept = regressor.intercept_
print(r2)
print(mae)
print(mse)
print(coef)
print(intercept)
print(rmse)
print(adjusted_r2)

pol = PolynomialFeatures(degree=2, include_bias=True) #also default vers
X_train_pol = pol.fit_transform(X_train)
X_test_pol = pol.transform(X_test)

regressor.fit(X_train_pol, y_train)
y_pred2 = regressor.predict(X_test_pol)
r22 = r2_score(y_test, y_pred2)
adjusted_r22 = 1 - (((1-r22)*(len(y_test) - 1))/ len(y_test)- X_test_pol.shape[1] - 1)
print(r22)
print(adjusted_r22)

new_dat = pd.read_csv("3-newdatas.csv")
new_dat.rename(columns={ "0": "Customer Satisfaction" }, inplace=True)
print(new_dat)

X_new_scaled = scaler.transform(new_dat)
X_new_scaled_pol = pol.transform(X_new_scaled)
y_pred_new_dat = regressor.predict(X_new_scaled_pol)

print(y_pred_new_dat)

fig = plt.figure()
plt.plot(new_dat, y_pred_new_dat, color="green", label="New Data")
plt.scatter(X_train, y_train, color="red", label="Training Data")
plt.scatter(X_test, y_test, color="blue", label="Test Data")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.legend()
plt.show()
def poly_reg (degree):
    lin = LinearRegression()
    poly = PolynomialFeatures(degree=degree)
    scalerr = StandardScaler()
    pipeline = Pipeline([
        ("scaler", scalerr),
        ("Poly_feat", poly),
        ("regressor", lin)
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print("sc: ", score)

    new_pred = pipeline.predict(new_dat)
    plt.plot(new_dat, new_pred, color="green", label="New Data")
    plt.scatter(X_train, y_train, color="red", label="Training Data")
    plt.scatter(X_test, y_test, color="blue", label="Test Data")
    plt.xlabel("Customer Satisfaction")
    plt.ylabel("Incentive")
    plt.legend()
    plt.show()

for degree in range(1, 8):
    poly_reg(degree)  #4.th degree seems best according to the score table
