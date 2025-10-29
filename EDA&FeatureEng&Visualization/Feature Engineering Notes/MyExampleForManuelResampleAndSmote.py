import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import numpy as np
from imblearn.over_sampling import SMOTE

np.random.seed(12)
set1 = 900
set2 = 100

#data balancing
#first, manual resampling
data1 = pd.DataFrame({
    "f1" : np.random.normal(loc=0, scale=1, size=set1),
    "f2" : np.random.normal(loc=0, scale=1, size=set1),
    "target" : [0] * set1,
})

data2 = pd.DataFrame({
    "f1" : np.random.normal(loc=0, scale=1, size=set2),
    "f2" : np.random.normal(loc=0, scale=1, size=set2),
    "target" : [1] * set2,
})

print(data1.info())
print(data2.info())
print(data1.describe())
print(data2.describe())

data = pd.concat([data1, data2]).reset_index(drop=True)

plt.figure(figsize= (5,3))
sns.scatterplot(x="f1", y="f2", hue="target", data=data)

down_sample_manuel = resample(data,
                              n_samples=set2,
                              replace=True,
                              random_state=12)
print(down_sample_manuel.info())
downsampled_merge = pd.concat([down_sample_manuel, data2],ignore_index=True)

up_sample_manuel = resample(data,
                            n_samples=set1,
                            random_state=12,
                            replace=True)
print(up_sample_manuel)
upsampled_merge = pd.concat([up_sample_manuel, data1], ignore_index=True)


#result for manual resampling
print(downsampled_merge.mean())
print(upsampled_merge.mean())
print(downsampled_merge.mean()-upsampled_merge.mean())

#resampling with SMOTE

resamplewithsmoth = SMOTE()
x,y = resamplewithsmoth.fit_resample(data[["f1", "f2"]],data["target"])
result = pd.concat([x,y])

print(result[result["target"] == 1].sum())
print(result[result["target"] == 0].sum())
print(result[result["target"] == 0])
print(result.mean())

plt.figure(figsize=[5,3])
sns.scatterplot(x="f1", y="f2", hue="target", data= result)
plt.show()