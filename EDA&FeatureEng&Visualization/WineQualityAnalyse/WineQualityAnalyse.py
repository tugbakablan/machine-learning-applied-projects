import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('WineQT.csv')

print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.duplicated().sum())

#grouping styles
print(data.groupby([data["quality"] == 3,"pH"]).mean())
print(data[data["quality"] == 3].groupby(["pH","alcohol"]).mean())

#get different graph styles
sns.heatmap(data.corr(), annot=True) #heatmap for correlation

graph = sns.pairplot(data[["alcohol","pH"]], height=3, aspect = 1)
graph.fig.suptitle("Relationship between alcohol and pH")

plt.figure()
sns.scatterplot(x="alcohol", y="pH", hue="quality", data=data)

plt.figure()
sns.boxplot(x="quality", y="alcohol", data=data)

data["quality"].plot(kind="hist", figsize=(4,4))
plt.xlabel("Quality")
plt.ylabel("Count")
plt.title("Histogram of qualşty")

column = data.columns.tolist()
(figure, axes) = plt.subplots(4,4, figsize=(12,12))
axes = axes.flatten()

for i, column in enumerate(column):
    sns.kdeplot(
        data = data,
        x = column,
        hue = data["quality"],
        ax = axes[i],
    )
axes[i].set_xlabel(f"{column} Distribution")
axes[i].set_ylabel(None)

for i in range(i+1 , len(axes)):
    axes[i].axis("off")

plt.show()