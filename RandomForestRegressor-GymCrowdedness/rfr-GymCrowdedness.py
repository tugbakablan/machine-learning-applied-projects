import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("gym_crowdedness.csv")

print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'], utc=True)
df['year'] = df['date'].dt.year
df.drop(['date'], axis=1, inplace=True)

sns.lineplot(data=df, x='hour', y='number_people', errorbar=None)
plt.title('Crowdedness by Hours')
plt.show()

sns.barplot(data=df, x='day_of_week', y='number_people')
plt.title('Crowdedness by Day')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.title('Corelation between Columns')
plt.show()

df.drop('timestamp', axis=1, inplace = True)

X = df.drop('number_people', axis = 1)
y = df['number_people']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

std_scaler = StandardScaler()
std_scaler.fit_transform(X_train,y_train)
std_scaler.transform(X_test)

lin = LinearRegression()
lin.fit(X_train, y_train)
lin.score(X_train, y_train)

def calculate_score(true, predicted):
    mae = mean_absolute_error(true, predicted),
    r2 = r2_score(true, predicted),
    mse = mean_squared_error(true, predicted),
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, r2, mse, rmse

models = {
    "Linear Regression" : LinearRegression(),
    "Ridge" : Ridge(),
    "Lasso" : Lasso(),
    "K-Neighbours Regressor" : KNeighborsRegressor(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    MAE, R2, MSE, RMSE = calculate_score(y_train, y_train_pred)
    xMAE, xR2, xMSE, xRMSE = calculate_score(y_test, y_test_pred)

    print(list(models.values())[i])
    print("Evaluation for Train DataSet")
    print("MAE: ", MAE)
    print("RMSE: ", RMSE)
    print("MSE: ", MSE)
    print("R2: ", R2)
    print("-------------------------------")
    print("Evaluation for Test DataSet")
    print("MAE: ", xMAE)
    print("RMSE: ", xRMSE)
    print("MSE: ", xMSE)
    print("R2: ", xR2)
    print("\n")

knn_params = {
    "n_neighbors": list(range(1, 11)),
}

rf_params = {
    "n_estimators" : [100,200,500,1000],
    "max_depth" : [2,5,7,10,15, None],
    "min_samples_split" : [2,8,12,20],
    "max_features": ["auto", "sqrt", "log2"]
}

randomcv_models = [
    ("KNN", KNeighborsRegressor(), knn_params),
    ("RandomForestRegressor", RandomForestRegressor(), rf_params)
]

for name , model, params in randomcv_models:
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=params,n_jobs=-1)
    randomcv.fit(X_train, y_train)
    print("best params for:", model, randomcv.best_params_)

models_chosen = {
    "K-Neighbours Regressor" : KNeighborsRegressor(n_neighbors = 2),
    "Random Forest Regressor": RandomForestRegressor(n_estimators= 1000, min_samples_split= 12, max_features='sqrt', max_depth= None)
}

for i in range(len(list(models_chosen))):
    modelx = list(models_chosen.values())[i]
    modelx.fit(X_train, y_train)

    y_train_pred = modelx.predict(X_train)
    y_test_pred = modelx.predict(X_test)

    MAE, R2, MSE, RMSE = calculate_score(y_train, y_train_pred)
    xMAE, xR2, xMSE, xRMSE = calculate_score(y_test, y_test_pred)

    print(list(models_chosen.values())[i])
    print("\nEvaluation for Train DataSet")
    print("MAE: ", MAE)
    print("RMSE: ", RMSE)
    print("MSE: ", MSE)
    print("R2: ", R2)
    print("-------------------------------")
    print("Evaluation for Test DataSet")
    print("MAE: ", xMAE)
    print("RMSE: ", xRMSE)
    print("MSE: ", xMSE)
    print("R2: ", xR2)
    print("\n")