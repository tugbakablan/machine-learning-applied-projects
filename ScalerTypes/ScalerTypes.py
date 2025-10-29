import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.read_csv("scalerdata.csv")
print(data.head())

fig , axes = plt.subplots(3,3)
axes = axes.flatten()
for i, column in enumerate(data.columns):
    sns.histplot(data = data[column], ax = axes[i], kde = True)
    axes[i].set_title(f" Original Version of {column}")

scaler = {
    "standard" : StandardScaler(),
    "minmax" : MinMaxScaler(),
    "robust" : RobustScaler()
}

scaler_data_frame = {}

for name, scaler in scaler.items():
    scaled = scaler.fit_transform(data)
    scaler_data_frame[name] = pd.DataFrame(scaled, columns= data.columns)

print(scaler_data_frame)

fig, axes = plt.subplots(1,3)
plt.subplots(1,3,1)

for i , column in enumerate(scaler_data_frame.items()):