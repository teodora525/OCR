from keras import layers, models, regularizers
from tensorflow import keras
from keras import layers

# model treniran za cifre
def build_cnn(input_shape=(28, 28, 1), num_classes: int = 10) -> models.Model:
    """
    CNN model za prepoznavanje cifara (MNIST).
    input_shape: oblik ulazne slike (28x28x1)
    num_classes: broj klasa (10 za cifre)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation (radi samo u trening modu)
    data_augmentation = models.Sequential(name="augmentation")
    data_augmentation.add(layers.RandomRotation(0.10))          # ~±10°
    data_augmentation.add(layers.RandomTranslation(0.10, 0.10)) # do 10% pomeraja
    data_augmentation.add(layers.RandomZoom(0.10))              # do 10% zum
    data_augmentation.add(layers.GaussianNoise(0.05))           # blagi šum

    x = data_augmentation(inputs)

    #CNN
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="mnist_cnn")
    return model

# model kreiran za slova
def build_letters_cnn(input_shape=(28, 28, 1), num_classes=26):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="emnist_letters_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model