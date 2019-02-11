import numpy as np

def power(sample1, sample2, reps, size, alpha):
    pvalue = 0
    pvalue_larger = 0
    t_obs = np.mean(sample2) - np.mean(sample1)

    for _ in range(reps):
        for i in range(reps):
            sample1_new = np.random.choice(sample1, sample1.size, replace=True)
            sample2_new = np.random.choice(sample2, sample2.size, replace=True)

            t_perm = np.mean(sample2_new) - np.mean(sample1_new)
            if t_perm > t_obs:
                pvalue = pvalue + 1
        pvalue = pvalue / reps
        if pvalue > alpha:
            pvalue_larger = pvalue_larger + 1

    return pvalue_larger/reps