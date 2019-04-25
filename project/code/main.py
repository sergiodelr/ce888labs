import pandas as pd
from scipy.io import arff

from preprocessing import preprocess_arrhythmia, preprocess_HTRU2, preprocess_phish
from training import train


if __name__ == '__main__':
    # Phishing dataset 
    phish_arff = arff.loadarff("../datasets/phishing.arff")
    df_phish = pd.DataFrame(phish_arff[0])
    print(df_phish.head())

    # HTRU 2 dataset
    df_h = pd.read_csv("../datasets/HTRU_2.csv", header=None)
    print(df_h.head())
    df_h.rename({8: "Class"}, axis="columns", inplace=True)

    # Arrhythmia dataset
    df_arr = pd.read_csv("../datasets/arrhythmia.csv", header=None)
    print(df_arr.head())
    df_arr.rename({279: "Class"}, axis="columns", inplace=True)

    # Preprocessing
    df_phish = preprocess_phish(df_phish)
    df_h = preprocess_HTRU2(df_h)
    df_arr = preprocess_arrhythmia(df_arr)

    # Train arrhythmia
    # Relu
    # arr_metrics_dict = {}
    # arr_metrics_noise_dict = {}
    # arr_loss = {}
    # arr_metrics_dict["256_1"], arr_loss["256_1"] = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [256], True, True)
    # arr_metrics_dict["128_1"], arr_loss["128_1"] = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [128], True, True)
    # arr_metrics_dict["256_200_1"], arr_loss["256_200_1"] = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 200], True, True)
    # arr_metrics_dict["256_128_1"], arr_loss["256_128_1"] = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 128], True, True)
    # arr_metrics_dict["128_64_1"], arr_loss["128_64_1"] = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [128, 64], True, True)
    # arr_metrics_dict["256_128_64_1"], arr_loss["256_128_64_1"] = train(df_arr, [0.3, 0.5, 0.7, 1], 3, [256, 128, 64], True, True)
    #
    # arr_metrics_noise_dict["256_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [256], True, True, True)
    # arr_metrics_noise_dict["128_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [128], True, True, True)
    # arr_metrics_noise_dict["256_200_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 200], True, True, True)
    # arr_metrics_noise_dict["256_128_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 128], True, True, True)
    # arr_metrics_noise_dict["128_64_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [128, 64], True, True, True)
    # arr_metrics_noise_dict["256_128_64_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 3, [256, 128, 64], True, True, True)
    #
    # arr_metrics_dict["256_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [256], False, True)
    # arr_metrics_dict["128_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [128], False, True)
    # arr_metrics_dict["256_200_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 200], False, True)
    # arr_metrics_dict["256_128_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 128], False, True)
    # arr_metrics_dict["128_64_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [128, 64], False, True)
    # arr_metrics_dict["256_128_64_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 3, [256, 128, 64], False, True)
    #
    # arr_metrics_df = pd.DataFrame.from_dict(arr_metrics_dict, orient='index',
    #                                         columns=['0.3', '0.5', '0.7', '1'])
    # arr_metrics_df.to_csv('../results/arrhythmia_16batch_relu_30test.csv')
    #
    # arr_metrics_noise_df = pd.DataFrame.from_dict(arr_metrics_noise_dict, orient='index',
    #                                         columns=['0.3', '0.5', '0.7', '1'])
    # arr_metrics_noise_df.to_csv('../results/arrhythmia_16batch_noise_30test.csv')
    #
    # arr_loss_df = pd.DataFrame.from_dict(arr_loss, orient='index',
    #                                         columns=['0.3', '0.5', '0.7', '1'])
    # arr_loss_df.to_csv('../results/arrhythmia_16batch_loss.csv')
    #
    # # Sigmoid
    # arr_metrics_dict = {}
    # arr_metrics_dict["256_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [256], True, False)
    # arr_metrics_dict["128_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [128], True, False)
    # arr_metrics_dict["256_200_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 200], True, False)
    # arr_metrics_dict["256_128_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 128], True, False)
    # arr_metrics_dict["128_64_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [128, 64], True, False)
    # arr_metrics_dict["256_128_64_1"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 3, [256, 128, 64], True, False)
    #
    # arr_metrics_dict["256_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [256], False, False)
    # arr_metrics_dict["128_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 1, [128], False, False)
    # arr_metrics_dict["256_200_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 200], False, False)
    # arr_metrics_dict["256_128_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [256, 128], False, False)
    # arr_metrics_dict["128_64_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 2, [128, 64], False, False)
    # arr_metrics_dict["256_128_64_0"], _ = train(df_arr, [0.3, 0.5, 0.7, 1], 3, [256, 128, 64], False, False)
    #
    # arr_metrics_df = pd.DataFrame.from_dict(arr_metrics_dict, orient='index',
    #                                         columns=['0.3', '0.5', '0.7', '1'])
    # arr_metrics_df.to_csv('../results/arrhythmia_16batch_sigmoid_30test.csv')

    # Train phishing
    # Relu
    phish_loss = {}
    phish_metrics_dict = {}
    phish_metrics_noise_dict = {}
    phish_metrics_dict["32_1"], phish_loss["32_1"] = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [32], True, True)
    phish_metrics_dict["24_1"], phish_loss["24_1"] = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [24], True, True)
    phish_metrics_dict["32_16_1"], phish_loss["32_16_1"] = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 16], True, True)
    phish_metrics_dict["32_24_1"], phish_loss["32_24_1"] = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 24], True, True)

    phish_metrics_noise_dict["32_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [32], True, True, True)
    phish_metrics_noise_dict["24_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [24], True, True, True)
    phish_metrics_noise_dict["32_16_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 16], True, True, True)
    phish_metrics_noise_dict["32_24_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 24], True, True, True)

    phish_metrics_dict["32_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [32], False, True)
    phish_metrics_dict["24_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [24], False, True)
    phish_metrics_dict["32_16_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 16], False, True)
    phish_metrics_dict["32_24_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 24], False, True)

    phish_metrics_df = pd.DataFrame.from_dict(phish_metrics_dict, orient='index',
                                              columns=['0.3', '0.5', '0.7', '1'])
    phish_metrics_df.to_csv('../results/phishing_16batch_relu_30test.csv')

    phish_metrics_noise_df = pd.DataFrame.from_dict(phish_metrics_noise_dict, orient='index',
                                              columns=['0.3', '0.5', '0.7', '1'])
    phish_metrics_noise_df.to_csv('../results/phishing_16batch_noise_30test.csv')

    phish_loss_df = pd.DataFrame.from_dict(phish_loss, orient='index',
                                         columns=['0.3', '0.5', '0.7', '1'])
    phish_loss_df.to_csv('../results/phishing_16batch_loss.csv')

    # Sigmoid
    phish_metrics_dict = {}
    phish_metrics_dict["32_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [32], True, False)
    phish_metrics_dict["24_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [24], True, False)
    phish_metrics_dict["32_16_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 16], True, False)
    phish_metrics_dict["32_24_1"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 24], True, False)

    phish_metrics_dict["32_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [32], False, False)
    phish_metrics_dict["24_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 1, [24], False, False)
    phish_metrics_dict["32_16_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 16], False, False)
    phish_metrics_dict["32_24_0"], _ = train(df_phish, [0.3, 0.5, 0.7, 1], 2, [32, 24], False, False)

    phish_metrics_df = pd.DataFrame.from_dict(phish_metrics_dict, orient='index',
                                              columns=['0.3', '0.5', '0.7', '1'])
    phish_metrics_df.to_csv('../results/phishing_16batch_sigmoid_30test.csv')

    # Train HTRU2
    # Relu
    htru_metrics_dict = {}
    htru_metrics_noise_dict = {}
    htru_loss ={}
    htru_metrics_dict["5_1"], htru_loss["5_1"] = train(df_h, [0.3, 0.5, 0.7, 1], 1, [5], True, True)
    htru_metrics_dict["4_1"], htru_loss["4_1"] = train(df_h, [0.3, 0.5, 0.7, 1], 1, [4], True, True)
    htru_metrics_dict["6_4_1"], htru_loss["6_4_1"] = train(df_h, [0.3, 0.5, 0.7, 1], 2, [6, 4], True, True)
    htru_metrics_dict["4_2_1"], htru_loss["4_2_1"] = train(df_h, [0.3, 0.5, 0.7, 1], 2, [4, 2], True, True)

    htru_metrics_noise_dict["5_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [5], True, True, True)
    htru_metrics_noise_dict["4_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [4], True, True, True)
    htru_metrics_noise_dict["6_4_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [6, 4], True, True, True)
    htru_metrics_noise_dict["4_2_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [4, 2], True, True, True)

    htru_metrics_dict["5_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [5], False, True)
    htru_metrics_dict["4_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [4], False, True)
    htru_metrics_dict["6_4_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [6, 4], False, True)
    htru_metrics_dict["4_2_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [4, 2], False, True)

    htru_metrics_df = pd.DataFrame.from_dict(htru_metrics_dict, orient='index',
                                             columns=['0.3', '0.5', '0.7', '1'])
    htru_metrics_df.to_csv('../results/htru2_16batch_relu_30test.csv')

    htru_metrics_noise_df = pd.DataFrame.from_dict(htru_metrics_noise_dict, orient='index',
                                             columns=['0.3', '0.5', '0.7', '1'])
    htru_metrics_noise_df.to_csv('../results/htru2_16batch_noise_30test.csv')

    htru_loss_df = pd.DataFrame.from_dict(htru_loss, orient='index',
                                         columns=['0.3', '0.5', '0.7', '1'])
    htru_loss_df.to_csv('../results/htru2_16batch_loss.csv')

    # Sigmoid
    htru_metrics_dict = {}
    htru_metrics_dict["5_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [5], True, False)
    htru_metrics_dict["4_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [4], True, False)
    htru_metrics_dict["6_4_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [6, 4], True, False)
    htru_metrics_dict["4_2_1"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [4, 2], True, False)

    htru_metrics_dict["5_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [5], False, False)
    htru_metrics_dict["4_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 1, [4], False, False)
    htru_metrics_dict["6_4_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [6, 4], False, False)
    htru_metrics_dict["4_2_0"], _ = train(df_h, [0.3, 0.5, 0.7, 1], 2, [4, 2], False, False)

    htru_metrics_df = pd.DataFrame.from_dict(htru_metrics_dict, orient='index',
                                             columns=['0.3', '0.5', '0.7', '1'])
    htru_metrics_df.to_csv('../results/htru2_16batch_sigmoid_30test.csv')
