import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('./vehicles.csv')
    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)
    sns_plot.savefig("scatterplot.png", bbox_inches='tight')

    plt.clf()
    data = df.values.T[1][:79]
    sns_plot2 = sns.distplot(data, bins=5, kde=False, rug=True).get_figure()
    sns_plot2.savefig("histogram.png", bbox_inches='tight')