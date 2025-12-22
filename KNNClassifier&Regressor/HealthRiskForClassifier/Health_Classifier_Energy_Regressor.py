import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

bmi_data = pd.read_csv('health_risk_classification.csv')
print(bmi_data.head())
print(bmi_data.describe())
print(bmi_data.info())

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x="blood_pressure_variation",
    y="activity_level_index",
    hue="high_risk_flag",
    data=bmi_data,
    ax=ax
)

ax.set_title("Health Risk")
ax.set_xlabel("Blood Pressure")
ax.set_ylabel("Activity Level Index")

imbalanced_or_not = bmi_data["high_risk_flag"].value_counts()
print(imbalanced_or_not)

plt.figure(figsize=(6,6))
sns.boxenplot(bmi_data)

X = bmi_data.drop(["high_risk_flag"], axis=1)
y = bmi_data["high_risk_flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print(accuracy_score(y_pred, y_test))

classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print(accuracy_score(y_pred, y_test))

energy = pd.read_csv('house_energy_regression.csv')
print(energy.head())
print(energy.describe())
print(energy.info())

plt.figure(figsize= [6,6])
sns.scatterplot( x = energy["outdoor_humidity_level"], y = energy["daily_energy_consumption_kwh"])
plt.figure(figsize= [6,6])
sns.scatterplot( x = energy["avg_indoor_temp_change"], y = energy["daily_energy_consumption_kwh"])

print(energy.corr())

X_1 = energy.drop(["daily_energy_consumption_kwh"], axis=1)
y_1 = energy["daily_energy_consumption_kwh"]
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=0.2, random_state=15)

scaler1 = StandardScaler()
X_1_train_scaled = scaler1.fit_transform(X_1_train)
X_1_test_scaled = scaler1.transform(X_1_test)
regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
regressor.fit(X_1_train_scaled, y_1_train)
y_pred_1 = regressor.predict(X_1_test_scaled)

print("MSE:", mean_squared_error(y_1_test, y_pred_1))
print("MAE:", mean_absolute_error(y_1_test, y_pred_1))
print("r2 :", r2_score(y_1_test, y_pred_1))

plt.show()