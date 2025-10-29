import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

data = sns.load_dataset('titanic')
print(data.columns.tolist())
print(data[["sibsp", "parch", "fare", "embarked", "class"]].head())

print(data[["sex", "class", "embark_town"]].isna().sum())
data = data.dropna(subset=['embark_town'])

encoder_one_hot = pd.get_dummies(data, columns =["sex","class","embark_town"], drop_first=True)
print(encoder_one_hot.columns.tolist())
#----------------------------------------------------------------------------------------------
data1 = data.copy()
encoder_label = LabelEncoder()

data1["sex"] = encoder_label.fit_transform(data1["sex"])
print(data1)
#----------------------------------------------------------------------------------------------
data2 = data.copy()
class_x = [["Third", "Second", "First"]]
encoder_ordinal = OrdinalEncoder(categories= class_x)
data2["class"] = encoder_ordinal.fit_transform(data2[["class"]])
print(data2)
#-----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1,3,figsize=(20,5))

data["sex"].value_counts().plot(kind="bar", ax=ax[0], title="Original")
encoder_one_hot["sex_male"].value_counts().plot(kind="bar", ax=ax[1], title="One-Hot Encoding")
data1["sex"].value_counts().plot(kind="bar", ax=ax[2], title="Label Encoder")

plt.show()