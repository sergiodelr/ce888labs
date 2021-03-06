from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from models import pretrained_encoder, encoder_classifier_model


def train(df, training_proportions, hidden_layers, layer_sizes, same_proportion_for_fine_tuning=True,
          autoencoder_reconstruct_input=False, add_noise=False):
    """
    Trains autoencoders with different amounts of training examples and returns each of their accuracies
    and ROC scores.
    :param df: Dataframe with data to be fitted. Must contain a column named 'Class' that corresponds to the
                class of the examples.
    :param training_proportions: List of the proportions of the dataset that will be used for training in decimal form.
    :param hidden_layers: Number of hidden layers of the encoder.
    :param layer_sizes: List of sizes of the hidden layers.
    :param same_proportion_for_fine_tuning: Whether or not to use the same proportion of the training data for the
                                            fine-tuning.  If set to True, the same
                                            proportion of data used for pre training will be used for fine-tuning and
                                            the rest for testing. If set to False, the fine-tuning will be carried
                                            out with 80% of the data, leaving 20% for testing. Default is True.
    :param autoencoder_reconstruct_input: Whether to train the autoencoder greedily to reconstruct the input or to
                                            reconstruct the previous layer. The default is False.
    :param add_noise: Whether to add noise to the training data to train a denoising autoencoder. Default is False.
    :return: List of tuples of the form (Accuracy, ROC score) for each of the amounts of training examples.
    """
    results = []
    encoder_losses = []
    x = df.drop(columns="Class")
    y = df["Class"]
    for proportion in training_proportions:
        if proportion != 1:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - proportion, random_state=1)
            if add_noise:
                x_noise_train = x_train + np.random.normal(0, 0.1, x_train.shape)
                x_noise_test = x_test + np.random.normal(0, 0.1, x_test.shape)
                encoder, loss = pretrained_encoder(x_train, x_test, hidden_layers, layer_sizes,
                                                   reconstruct_input=autoencoder_reconstruct_input,
                                                   x_noise_train=x_noise_train, x_noise_test=x_noise_test)
            else:
                encoder, loss = pretrained_encoder(x_train, x_test, hidden_layers, layer_sizes,
                                                   reconstruct_input=autoencoder_reconstruct_input)
        else:
            if add_noise:
                x_noise_train = x + np.random.normal(0, 0.1, x.shape)
                encoder, loss = pretrained_encoder(x, None, hidden_layers, layer_sizes,
                                                   reconstruct_input=autoencoder_reconstruct_input,
                                                   x_noise_train=x_noise_train)
            else:
                encoder, loss = pretrained_encoder(x, None, hidden_layers, layer_sizes,
                                                   reconstruct_input=autoencoder_reconstruct_input)
        model = encoder_classifier_model(encoder)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(f'\n\nTraining model:\
                Hidden layers: {hidden_layers} Layer sizes: {layer_sizes} Training proportion: {proportion}')
        if same_proportion_for_fine_tuning and proportion != 1:
            x_train_fine = x_train
            y_train_fine = y_train
            x_test_fine = x_test
            y_test_fine = y_test
        else:
            x_train_fine, x_test_fine, y_train_fine, y_test_fine = train_test_split(x, y, test_size=.3, random_state=1)

        early_stopping = EarlyStopping('val_accuracy', patience=20)
        model.fit(x_train_fine, y_train_fine,
                  batch_size=16,
                  epochs=50,
                  verbose=1,
                  callbacks=[early_stopping],
                  validation_data=(x_test_fine, y_test_fine))
        # Evaluate model
        metrics = model.evaluate(x_test_fine, y_test_fine, batch_size=16)
        y_auc = model.predict(x_test_fine, batch_size=16)
        auc = roc_auc_score(y_test_fine, y_auc)
        metrics.append(auc)
        encoder_losses.append(loss)

        print(f"Metrics: {metrics}")
        results.append(metrics)

    return results, encoder_losses
