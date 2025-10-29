import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data= pd.read_csv("Algerian_forest_fires_dataset.csv")
print(data.columns.tolist())
print(data.shape)
print(data.info())
print(data.head())
print(data.isnull().sum())
print(data[data.isnull().any(axis=1)])

data.drop(122, inplace= True)
data.loc[:123,'Region'] = 0
data.loc[123:,'Region'] = 1

print(data["day"].unique())
print(data[data["day"] == "day"], data[data["day"] == "Sidi-Bel Abbes Region Dataset"])
data.drop([123,124], inplace=True)
data.columns.str.strip()

print(data.columns)
data[['day','month','year','Temperature',' RH',' Ws']] = data[['day','month','year','Temperature',' RH',' Ws']].astype(int)

print(data["FWI"].unique())
print(data[data["FWI"] == "fire   "])
data.loc[168, 'Classes  '] = 'fire'
data["FWI"] = pd.to_numeric(data["FWI"], errors='coerce')
data.loc[168, 'FWI'] = data["FWI"].mean()
data[['Rain ','FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']] = data[['Rain ','FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float)
print(data.info())

print(data["Classes  "].value_counts())
data["Classes  "] = np.where(data["Classes  "].str.contains("not fire"), 1, 0)
print(data["Classes  "].value_counts())
print(data["Classes  "].value_counts(normalize=True)*100)

print(data.corr())

sns.heatmap(data.corr(), annot= True, cmap= 'coolwarm')
plt.show()

X = data.drop(["FWI"], axis = 1)
y = data["FWI"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)
print(X_train.corr())

def remove_extra_corred(data, value_to_compare):
    removed_ones = set()
    corr = data.corr()
    for value in range(len(corr.columns)):
        for i in range(value):
            if abs(corr.iloc[value, i]) > value_to_compare:
                removed_ones.add(corr.columns[value])
    return removed_ones

cittu = remove_extra_corred(X_train, 0.80)
print(cittu)
X_train.drop(columns = cittu, axis=1, inplace=True)
X_test.drop(columns = cittu, axis=1, inplace=True)

print(X_train.columns)
print(X_test.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.subplots(figsize = (12,5))
plt.subplot(1,2,1)
sns.boxplot(data = X_train)
plt.subplot(1,2,2)
sns.boxplot(data = X_train_scaled)
plt.show()

linear = LinearRegression()
linear.fit(X_train_scaled,y_train)
y_prediction = linear.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)

plt.scatter(y_test, y_prediction)
plt.show()

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_prediction_ridge = ridge.predict(X_test_scaled)

lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_prediction_lasso = lasso.predict(X_test_scaled)

from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet()
elastic_net.fit(X_train_scaled, y_train)
y_prediction_elastic_net = elastic_net.predict(X_test_scaled)
plt.subplots(figsize = (12,5))
plt.subplot(1,4,1)
plt.scatter(y_test, y_prediction)
plt.subplot(1,4,2)
plt.scatter(y_test, y_prediction_ridge)
plt.subplot(1,4,3)
plt.scatter(y_test, y_prediction_lasso)
plt.subplot(1,4,4)
plt.scatter(y_test, y_prediction_elastic_net)
plt.title("With Other Options Predicted")

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV

lasso_cv = LassoCV()
lasso_cv.fit(X_train_scaled, y_train)
y_prediction_lasso_cv = lasso_cv.predict(X_test_scaled)

ridge_cv = RidgeCV()
ridge_cv.fit(X_train_scaled, y_train)
y_prediction_ridge_cv = ridge_cv.predict(X_test_scaled)

elastic_cv = ElasticNetCV()
elastic_cv.fit(X_train_scaled, y_train)
y_prediction_elastic_cv = elastic_cv.predict(X_test_scaled)

plt.subplots(figsize = (12,5))
plt.subplot(1,3,1)
plt.scatter(y_test, y_prediction_ridge_cv)
plt.subplot(1,3,2)
plt.scatter(y_test, y_prediction_lasso_cv)
plt.subplot(1,3,3)
plt.scatter(y_test, y_prediction_elastic_cv)
plt.xlabel(" Y Test ")
plt.ylabel(" Y Prediction ")
plt.title("Cross Validation Changed")

plt.show()