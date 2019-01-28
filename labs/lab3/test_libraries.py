import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sklearn as sk
from sklearn import datasets
import pandas as pd
cancer = datasets.load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

ids = range(569)
df["id"] = list(ids)
df.head()
df.describe()

sns.countplot(x="target", data=cancer)
plt.show()
sns.distplot(df["worst area"])
plt.show()
sns.pairplot(df)
plt.show()