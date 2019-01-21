import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('./vehicles.csv')
    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)
    sns_plot.savefig("scaterplot.png", bbox_inches='tight')