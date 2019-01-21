import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bootstrap as bs

def power(sample1, sample2, reps, size, alpha):
    pvalue_larger = 0

    t_obs = np.mean(sample2) - np.mean(sample1)

    sample = np.concatenate((sample1, sample2))
    for i in range(reps):
        perm = np.random.permutation(sample)
        new = np.split(perm, [size])
        t_perm = np.mean(new[1]) - np.mean(new[0])
        if t_perm > t_obs:
            pvalue_larger = pvalue_larger + 1

    return pvalue_larger/reps

if __name__ == '__main__':
    df = pd.read_csv('./vehicles.csv')
    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)
    sns_plot.savefig("scatterplot.png", bbox_inches='tight')

    plt.clf()

    axes = plt.gca()
    axes.set_xlabel('Millons of pounds in sales')
    axes.set_ylabel('Sales count')

    data = df.values.T[1][:79]
    sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()
    axes = plt.gca()
    axes.set_xlabel('MPG of new vehicles')
    axes.set_ylabel('Vehicle count')
    sns_plot2.savefig("histogramNew.png", bbox_inches='tight')
    sns_plot2.savefig("histogramNew.pdf", bbox_inches='tight')

    plt.clf()
    data_old = df.values.T[0]
    sns_plot3 = sns.distplot(data_old, bins=20, kde=False, rug=True).get_figure()
    axes = plt.gca()
    axes.set_xlabel('MPG of old vehicles')
    axes.set_ylabel('Vehicle count')
    sns_plot2.savefig("histogramOld.png", bbox_inches='tight')
    sns_plot2.savefig("histogramOld.pdf", bbox_inches='tight')

    old = np.array(data_old)
    new = np.array(data)
    print("Mean of old cars: %f" % np.mean(old))
    print("Mean of new cars: %f" % np.mean(new))

    old_bs_mean, old_bs_lower, old_bs_upper = bs.boostrap(old, old.shape[0], 100000, 95)
    new_bs_mean, new_bs_lower, new_bs_upper = bs.boostrap(new, new.shape[0], 100000, 95)
    print("Old lower: %f, Old mean: %f, Old upper: %f" % (old_bs_lower, old_bs_mean, old_bs_upper))
    print("New lower: %f, New mean: %f, New upper: %f" % (new_bs_lower, new_bs_mean, new_bs_upper))