from keras.layers import Input, Dense
from keras.models import Model


def pretrained_encoder(x_train, x_test, hidden_layers, layer_sizes, reconstruct_input=False):
    """
    Creates a pre trained encoder with the specified training data. The encoder is trained in a greedy layer-wise
    manner.
    :param reconstruct_input: Whether to train the autoencoder greedily to reconstruct the input or to reconstruct
                                the previous layer. Default is False.
    :param x_train: Training data (unsupervised) np array.
    :param x_test: Testing data (unsupervised) np array.
    :param hidden_layers: Number of hidden layers.
    :param layer_sizes: List of layer sizes.
    :return: Pre trained Keras model and its loss.
    """
    print(f"\nPre training model: Hidden layers: {hidden_layers} Layer sizes: {layer_sizes} Training data: {len(x_train)}")

    if reconstruct_input:
        activation = "relu"
    else:
        activation = "sigmoid"

    # Train first encoder layer
    input_layer = Input(shape=(x_train.shape[1],))
    encoder_layer = Dense(layer_sizes[0], activation=activation)(input_layer)
    decoder_layer = Dense(x_train.shape[1], activation="sigmoid")(encoder_layer)
    autoencoder = Model(input_layer, decoder_layer)
    encoder = Model(input_layer, encoder_layer)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    if x_test is not None:
        autoencoder.fit(x_train, x_train,
                        epochs=30,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(x_test, x_test))
    else:
        autoencoder.fit(x_train, x_train,
                        epochs=30,
                        batch_size=16,
                        shuffle=True)
    # Train rest of layers greedily
    for i in range(1, hidden_layers):
        # Generate training examples for layer i from latent features
        if reconstruct_input:
            x_train_latent = x_train
            if x_test is not None:
                x_test_latent = x_test
        else:
            x_train_latent = encoder.predict(x_train)
            if x_test is not None:
                x_test_latent = encoder.predict(x_test)

        # Freeze layers
        for layer in autoencoder.layers:
            layer.trainable = False

        # Add new encoder layer
        encoder_layer = Dense(layer_sizes[i], activation=activation)(encoder_layer)
        # Add dummy decoder layer to train encoder
        if reconstruct_input:
            decoder_layer = Dense(x_train.shape[1], activation="sigmoid")(encoder_layer)
        else:
            decoder_layer = Dense(layer_sizes[i - 1], activation="sigmoid")(encoder_layer)
        autoencoder = Model(input_layer, decoder_layer)
        encoder = Model(input_layer, encoder_layer)

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        if x_test is not None:
            autoencoder.fit(x_train, x_train_latent,
                            epochs=30,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(x_test, x_test_latent))
        else:
            autoencoder.fit(x_train, x_train_latent,
                            epochs=30,
                            batch_size=16,
                            shuffle=True)

    if reconstruct_input:
        if x_test is not None:
            metric = autoencoder.evaluate(x_test, x_test_latent)
        else:
            metric = autoencoder.evaluate(x_train, x_train_latent)
    else:
        metric = None
    return encoder, metric


def encoder_classifier_model(pretrained_encoder_model):
    """
    Creates a binary classifier model from the pre trained encoder.
    :param pretrained_encoder_model: The pre trained encoder.
    :param number_of_classes: The number of classes.
    :return: A Keras model consisting of the pre trained encoder followed by a sigmoid classifier.
    """
    # input_layer = pretrained_encoder_model.layers[0]
    last_layer = pretrained_encoder_model.layers[-1].output

    pretrained_encoder_model.summary()

    classifier_layer = Dense(1, activation="sigmoid")(last_layer)

    encoder_classifier = Model(pretrained_encoder_model.input, classifier_layer)
    return encoder_classifier
