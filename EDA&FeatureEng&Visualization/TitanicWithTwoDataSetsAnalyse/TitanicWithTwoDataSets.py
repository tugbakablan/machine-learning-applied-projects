import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')
df = sns.load_dataset('titanic')

print(data.info())
print(df.info())

print(data["Passengerid"].duplicated().sum())

map_dict = {
    "female" : 1,
    "male" : 0,
}

df = df.replace(map_dict)
# data = data.rename(columns = {"Age" : "age" , "Sex" : "sex"}) this is an ooption for several changed needed versions
data.columns = data.columns.str.strip().str.lower()
df.columns = df.columns.str.strip().str.lower()


common = data.columns.intersection(df.columns)
mergedData = pd.merge(data[common], df[common], how= "inner", on= "age")
print(mergedData.info())

plt.figure(figsize = (5,5))
sns.heatmap(mergedData.corr(numeric_only=True), annot=True) #no readable data ,but wanted see

plt.figure(figsize = (5,5))
sns.scatterplot(x="age", y="fare_x", hue="pclass_x", data = mergedData)
plt.figure(figsize = (5,5))
sns.histplot(data = mergedData, kde = True)

mergedAndFilled = mergedData.copy()
mergedAndFilled["embarked_x"] = mergedData["embarked_x"].fillna(mergedData["embarked_x"].mode()[0])
print(mergedAndFilled.head())

plt.figure(figsize = (5,5))
sns.kdeplot(x="age", y="fare_x", data=mergedAndFilled, fill=True)
plt.figure(figsize = (5,5))
sns.pairplot(mergedAndFilled, hue="age")

plt.show()


