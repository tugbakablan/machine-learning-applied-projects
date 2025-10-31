import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px

data = pd.read_csv('loan_risk_svm.csv')
print(data.info())
print(data["loan_risk"].value_counts())

sns.scatterplot(x=data["credit_score_fluctuation"], y=data["recent_transaction_volume"], hue = data["loan_risk"])

X= data.drop("loan_risk", axis=1)
y = data["loan_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_svc = SVC(kernel = "linear")
linear_svc.fit(X_train, y_train)
y_prediction = linear_svc.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_prediction, y_test))
print("classification report \n", classification_report(y_prediction, y_test))

rbf_svc = SVC(kernel = "rbf")
rbf_svc.fit(X_train, y_train)
y_prediction_rbf = rbf_svc.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_prediction_rbf, y_test))
print("classification report \n", classification_report(y_prediction_rbf, y_test))

poly_svc = SVC(kernel = "poly")
poly_svc.fit(X_train, y_train)
y_prediction_poly = poly_svc.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_prediction_poly, y_test))
print("classification report \n", classification_report(y_prediction_poly, y_test))

sigmoid_svc = SVC(kernel = "sigmoid")
sigmoid_svc.fit(X_train, y_train)
y_prediction_sigmoid = sigmoid_svc.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_prediction_sigmoid, y_test))
print("classification report \n", classification_report(y_prediction_sigmoid, y_test))

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))
print("classification report \n", classification_report(y_pred, y_test))

params = {  'C' : [100, 10, 1, 0.1, 0.01],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'kernel' : ['rbf']}

grid = GridSearchCV(estimator= SVC(), param_grid = params, cv = 5, refit = True)
grid.fit(X_train, y_train)
pred_y = grid.predict(X_test)
print("confusion matrix: \n", confusion_matrix(pred_y, y_test))
print("classification report \n", classification_report(pred_y, y_test))

print(grid.best_params_)
print(grid.best_score_)

data["credit_score_fluctuation**2"] = data["credit_score_fluctuation"]**2
data["recent_transaction_volume**2"] = data["recent_transaction_volume"]**2
data["credit_score_fluctuation*recent_transaction_volume"] = (data["recent_transaction_volume"]* data["credit_score_fluctuation"])

fig = px.scatter_3d(x=data["credit_score_fluctuation**2"],
                    y=data["recent_transaction_volume**2"],
                    z = data["credit_score_fluctuation*recent_transaction_volume"],
                    color = data["loan_risk"])
fig.show()

linear=SVC(kernel='linear')
linear.fit(X_train,y_train)
y_predd=linear.predict(X_test)
print(classification_report(y_test,y_predd))
print(confusion_matrix(y_test,y_predd))

rbf=SVC(kernel='rbf')
rbf.fit(X_train,y_train)
y_preddd=rbf.predict(X_test)
print(classification_report(y_test,y_preddd))
print(confusion_matrix(y_test,y_preddd))