import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

df_check = pd.read_csv("income_evaluation.csv")
print(df_check.columns.tolist())
print(df_check.info())

col_names = ['age', 'workclass', 'finalweight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df_check.columns = col_names
print(df_check.describe())

categorical = [col for col in df_check.columns if df_check[col].dtype == "O"]
numeric = [col for col in df_check.columns if df_check[col].dtype != "O"]

for col in categorical:
    print(df_check[col].value_counts())

for col in numeric:
    print(df_check[col].value_counts())

fig1, ax1 = plt.subplots(figsize= (6,6))
ax1 = sns.countplot(x="income", hue = "sex", data = df_check)
ax1.set_title("Distribution of income by gender")
plt.show()

fig, ax = plt.subplots(figsize= (6,6))
ax = sns.countplot(x="income", hue = "race", data = df_check)
ax.set_title("Distribution of income by race")
plt.show()
#men earn more than women, white people earn more than other races

print(df_check.head())
print(df_check["workclass"].unique())

df_check["workclass"] = df_check["workclass"].replace(' ?', np.nan)
df_check["occupation"] = df_check["occupation"].replace(' ?', np.nan)
df_check["native_country"] = df_check["native_country"].replace(' ?', np.nan)

sns.pairplot(df_check, hue="income")
plt.show()

X = df_check.drop("income", axis=1)
y = df_check["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 12)
for i in [X_train, X_test]:
    i["workclass"] = i["workclass"].fillna(X_train['workclass'].mode()[0])
    i["occupation"] = i["occupation"].fillna(X_train['occupation'].mode()[0])
    i["native_country"] = i["native_country"].fillna(X_train['native_country'].mode()[0])

print(X_train.isnull().sum())
print(df_check[categorical].nunique())
y_train_binary = y_train.apply(lambda x: 1 if x.strip() == ">50K" else 0 )
target_means = y_train_binary.groupby(X_train["native_country"]).mean()
print(target_means)

X_train['native_country_encoded'] = X_train['native_country'].map(target_means)
X_train['native_country_encoded'] = X_train['native_country_encoded'].fillna(y_train_binary.mean())
X_train = X_train.drop("native_country", axis=1)

X_test['native_country_encoded'] = X_test['native_country'].map(target_means)
X_test['native_country_encoded'] = X_test['native_country_encoded'].fillna(y_train_binary.mean())
X_test = X_test.drop("native_country", axis=1)

one_hot_categories = ['workclass',
 'education',
 'marital_status',
 'occupation',
 'relationship',
 'race',
 'sex']

encoder = ColumnTransformer (
    transformers = [('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_categories)], remainder = "passthrough"
)

X_train_enc = encoder.fit_transform(X_train)
X_test_enc = encoder.transform(X_test)
columns = encoder.get_feature_names_out()
X_train = pd.DataFrame(X_train_enc, columns = columns, index = X_train.index)
X_test = pd.DataFrame(X_test_enc, columns = columns, index = X_test.index)

rfc = RandomForestClassifier (n_estimators = 100, random_state = 12)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

feature_score = pd.Series(rfc.feature_importances_, index = X_train.columns).sort_values(ascending= False)
X_train = X_train.drop(['categorical__education_ 1st-4th', 'categorical__occupation_ Priv-house-serv', 'categorical__education_ Preschool', 'categorical__workclass_ Without-pay', 'categorical__occupation_ Armed-Forces', 'categorical__workclass_ Never-worked'], axis=1)
X_test = X_test.drop(['categorical__education_ 1st-4th', 'categorical__occupation_ Priv-house-serv', 'categorical__education_ Preschool', 'categorical__workclass_ Without-pay', 'categorical__occupation_ Armed-Forces', 'categorical__workclass_ Never-worked'], axis=1)

rfc2 = RandomForestClassifier (n_estimators = 100, random_state = 12)
rfc2.fit(X_train, y_train)
y_pred2 = rfc2.predict (X_test)
print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

params = {
    "n_estimators" : [100,200,500, 900],
    "max_depth" : [5,8,10,15, None],
    "max_features" : ["sqrt", "log2", 7,8],
    "min_samples_split" : [2, 8, 15, 20]
}

rscv = RandomizedSearchCV( estimator = RandomForestClassifier(), param_distributions = params, cv = 3, n_jobs = -1)
rscv.fit(X_train, y_train)
y_pred3 = rscv.predict(X_test)
print(rscv.best_params_)
print(accuracy_score(y_test, y_pred3))
print(classification_report(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))