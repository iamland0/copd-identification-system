import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# path ke data hasil ekstraksi ciri
DATA_PATH = "data_5.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # merubah lists ke numpy array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data Siap!")

    return  X, y


if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # membagi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # membuat arsitektur JST
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # hidden layer ke-1
        keras.layers.Dense(512, activation='relu'),

        # hidden layer ke-2
        keras.layers.Dense(256, activation='relu'),

        # hidden layer ke-3
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(2, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)






