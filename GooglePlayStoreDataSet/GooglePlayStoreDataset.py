import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
#import warnings
#warnings.filterwarnings("ignore")

data = pd.read_csv('googleplaystore.csv')
print(data.info())

print(data.isnull().sum())
print(data["Reviews"].str.isnumeric().sum())
print(data[~data["Reviews"].str.isnumeric()])
print(data["Reviews"].loc[10472])
data["Reviews"] = data["Reviews"].loc[10472].replace("M","")
data["Reviews"] = data["Reviews"].astype(float)
print(data["Reviews"].info())

column = ["Installs", "Size", "Price"]
symbols = ["+",",","$"]

for col in column:
    for sym in symbols:
        data[col] = data[col].str.replace(sym,"")
data["Installs"] = data["Installs"].replace("Free",np.nan)
data["Price"] = data["Price"].replace("Everyone",np.nan)
print(data.Size.unique())
data["Size"] = data["Size"].str.replace("M", "000")
data["Size"] = data["Size"].str.replace("k", "")
data["Size"] = data["Size"].replace("Varies with device",np.nan)
data["Size"] = data["Size"].astype(float)
data["Installs"] = data["Installs"].astype(float)
data["Price"] = data["Price"].astype(float) / 10000

print(data.info())

data["Last Updated"] = data["Last Updated"].str.strip()
data["Last Updated"] = pd.to_datetime(data["Last Updated"], format="mixed", errors="coerce")

print(data.info())

x = LabelEncoder()
data["Type"] = x.fit_transform(data["Type"])

data.drop_duplicates(subset= "App", keep="first")

categorical_column = [column for column in data.columns if data[column].dtype == 'O']
numerical_column = [column for column in data.columns if data[column].dtype != 'O']

plt.figure(figsize=[14,8])
for i in range(0, len(numerical_column)):
    plt.subplot(3,3,i+1)
    sns.kdeplot(data[numerical_column[i]], fill=True, color="k")
    plt.xlabel(numerical_column[i])
    plt.tight_layout()

plt.figure(figsize=[12,4])
category = ["Type", "Content Rating"]
for i in range(0, len(category)):
    plt.subplot(1,2, i+1)
    sns.countplot(x=category[i], data=data, color="m")
    plt.xlabel(category[i])
    plt.tight_layout()

data["Installs"] = data["Installs"]/100000
df = data.groupby(["Category", "App"])["Installs"].sum().reset_index()
df = df.sort_values("Installs", ascending=False)

apps = ['GAME',"COMMUNICATION","TOOLS","PRODUCTIVITY","SOCIAL"]
plt.figure(figsize = (30,20))
for i, app in enumerate(apps):
    df1 = df[df["Category"] == app]
    df1 = df1.head(5)

    plt.subplot(2,3, i+1)
    sns.barplot(data = df1, x="Installs", y = "App")
    plt.title(app, size = 12)
plt.show()

data["Android Ver"] = data["Android Ver"].replace("", np.nan, regex=True)
data["Android Ver"] = data["Android Ver"].replace("and up", "",regex=True)
data["Android Ver"] = data["Android Ver"].replace("Varies with device", "", regex =True)
data = data[~data["Android Ver"].astype(str).str.contains("-")]
#data = data[data["Android Ver"].str.contains("-") == False]

mean_genre = data.groupby("Genres")["Installs"].mean() / 100000
mean_genre = mean_genre.to_dict()
data["Genres Target Encoded"] = data["Genres"].map(mean_genre)