from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Phishing
phish_arff = arff.loadarff("../datasets/phishing.arff")
df_phish = pd.DataFrame(phish_arff[0])
print(df_phish.head())

p_countplot = sns.countplot(df_phish["Result"])
fig = p_countplot.get_figure()
fig.savefig("../figures/phishing_countplot.png")
plt.show()

# HTRU 2 dataset
df_h = pd.read_csv("../datasets/HTRU_2.csv", header=None)
print(df_h.head())
df_h.rename({8: "Class"}, axis="columns", inplace=True)

h_countplot = sns.countplot(df_h["Class"])
fig = h_countplot.get_figure()
fig.savefig("../figures/htru2_countplot.png")
plt.show()

h_scatterplot1 = sns.scatterplot(1, 2, data=df_h, hue="Class").set_title("(a)")
fig = h_scatterplot1.get_figure()
fig.savefig("../figures/htru2_scatterplot1.png")
plt.show()
h_scatterplot1 = sns.scatterplot(2, 7, data=df_h, hue="Class").set_title("(b)")
fig = h_scatterplot1.get_figure()
fig.savefig("../figures/htru2_scatterplot2.png")
plt.show()

# Arrhythmia dataset
df_arr = pd.read_csv("../datasets/arrhythmia.csv", header=None)
print(df_arr.head())
df_arr.rename({279: "Class"}, axis="columns", inplace=True)

arr_countplot = sns.countplot(df_arr["Class"])
fig = arr_countplot.get_figure()
fig.savefig("../figures/arr_countplot.png")
plt.show()

df_arr_dummy = pd.get_dummies(df_arr["Class"])
df_arr_dummy["Class"] = df_arr_dummy[1]
arr_countplot2 = sns.countplot(df_arr_dummy["Class"])
fig = arr_countplot2.get_figure()
fig.savefig("../figures/arr_countplot_binary.png")
plt.show()
