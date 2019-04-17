import pandas as pd
from sklearn import preprocessing

def preprocess_phish(df):
    # One hot encoding
    df = pd.get_dummies(df)
    del df["Result_b'-1'"]
    df.rename(columns={"Result_b'1'": "Class"}, inplace=True)

    print("Phishing ok")
    return df


def preprocess_HTRU2(df):
    x = df.values[:, :-1]
    y = df.values[:, -1]

    # Scale data from 0 to 1
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    df["Class"] = pd.Series(y)
    df = pd.get_dummies(df, columns=["Class"])
    del df["Class_0.0"]
    df.rename(columns={"Class_1.0": "Class"}, inplace=True)

    print("HTRU2 ok")
    return df


def preprocess_arrhythmia(df):
    # Delete missing values
    del df[13]
    df = df[(df[10] != '?') & (df[11] != '?') & (df[12] != '?') & (df[14] != '?')]

    df[10] = pd.to_numeric(df[10])
    df[11] = pd.to_numeric(df[11])
    df[12] = pd.to_numeric(df[12])
    df[14] = pd.to_numeric(df[14])

    # Preprocess linear and nominal values separately
    x_linear = df.drop(columns=[1, 20, 21, 22, 23, 24, 25]).values[:, :-1]
    x_nominal = df.iloc[:, [1, 20, 21, 22, 23, 24, 25]]
    y = df.iloc[:, -1]

    scaler = preprocessing.MinMaxScaler()
    x_linear = scaler.fit_transform(x_linear)
    x_nominal.reset_index(inplace=True, drop=True)

    y = pd.get_dummies(y)
    y = y[1]
    y.reset_index(inplace=True, drop=True)

    # Merge data into single data frame
    df = pd.DataFrame(x_linear)
    df = pd.concat([df, x_nominal], axis='columns', ignore_index=True)
    df['Class'] = y

    print("Arrhythmia ok")
    return df
