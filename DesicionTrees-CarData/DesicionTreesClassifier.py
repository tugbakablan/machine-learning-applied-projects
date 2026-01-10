import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("car_evaluation.csv")
print(data.head())

new_columns = [ "buying", "maintaince", "doors", "peopleCapacity", "lug_size", "safety", "class" ]
data.columns = new_columns
print(data.head())

for i in data.columns:
    print(data[new_columns].value_counts())

print(data.info())

print(data['doors'].unique())
print(data['peopleCapacity'].unique())
print(data['buying'].unique())
print(data['safety'].unique())

data['doors'] = data['doors'].replace('5more', '5').astype('int')
data['peopleCapacity'] = data['peopleCapacity'].replace('more', '5').astype('int')
print(data.info())

plt.figure(figsize=(10,10))
sns.boxplot(x = data['buying'], y = data['maintaince'], hue=data['class'])
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(x = data['buying'], hue=data['class'])
plt.show()

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
print(X_train.shape)

categorical_cols = [ "buying", "maintaince", "lug_size", "safety"]
numeric_cols = ["doors", "peopleCapacity"]

for col in categorical_cols:
    print(data[col].value_counts())

ordinal = OrdinalEncoder( categories = [
    ["low", "med", "high", "vhigh"], #for buying
    ["low", "med", "high", "vhigh"], #for maintaince
    ["small", "med", "big"], # for lugsize
    ["low", "med", "high"] #for safety
])

trans = ColumnTransformer( transformers=[
    ('ordinal', ordinal, categorical_cols)] , remainder = "passthrough"
)

X_train_trans = trans.fit_transform(X_train)
X_test_trans = trans.transform(X_test)
print(X_train_trans.shape)
print(X_test_trans.shape)

#random selection here, then checked by gridsearch
tree_des = DecisionTreeClassifier( criterion = "gini", max_depth = 5 , random_state = 12 )
tree_des.fit(X_train_trans, y_train)
y_pred = tree_des.predict(X_test_trans)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10,20))
colunms = categorical_cols + numeric_cols
tree.plot_tree(tree_des.fit(X_train_trans, y_train), feature_names = colunms, filled=True)
plt.show()

#hyperparametre tuning

param = {
    "criterion" : ["gini", "entropy", "log_loss"],
    "splitter" : ["best", "random"],
    "max_depth" : [1,2,3,4,5,15,None],
    "max_features" : ["sqrt", "log2", None]
}

grid = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=param, cv=5,scoring="accuracy")
grid.fit(X_train_trans, y_train)
print(grid.best_params_)
y_pred_new = grid.predict(X_test_trans)
print(accuracy_score(y_test, y_pred_new))
print(confusion_matrix(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))

tree_model_new = DecisionTreeClassifier(criterion="entropy",max_depth=None, max_features=None, splitter="best")
tree_model_new.fit(X_train_trans, y_train)
y_pred_new_1 = tree_model_new.predict(X_test_trans)
print(confusion_matrix(y_test, y_pred_new_1))
print(classification_report(y_test, y_pred_new_1))
print(accuracy_score(y_test, y_pred_new_1))

plt.figure(figsize=(90,90))
tree.plot_tree(tree_model_new.fit(X_train_trans, y_train), feature_names = colunms, filled=True)
plt.show()

df = pd.read_csv("C:/Users/tugba/Desktop/ML-Data/NaiveBayesClassifier-IrisSpecies/Iris.csv")
print(df.head())

X_1 = df.drop(['Id', 'Species'], axis = 1)
y_1 = df['Species']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=12)
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_1, y_train_1)
y_pred_purining = tree_model.predict(X_test_1)

plt.figure(figsize=(10,10))
tree.plot_tree(tree_model.fit(X_train_1, y_train_1), feature_names = X_train_1.columns, filled=True)
plt.show()

print(confusion_matrix(y_test_1, y_pred_purining))
print(classification_report(y_test_1, y_pred_purining))
print(accuracy_score(y_test_1, y_pred_purining))