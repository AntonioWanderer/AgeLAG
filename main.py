import os, cv2, numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, AveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_images(path: str = "LAG/"):
    x = []
    y = []
    folders = os.listdir(path)
    for folder in folders:
        imx = []
        imy = []
        cont = os.listdir(path + folder)
        for item in cont:
            if item == "y":
                for ys in os.listdir(path + folder + "/y/"):
                    imy.append(cv2.imread(path + folder + "/y/" + ys))
            else:
                imx.append(cv2.imread(path + folder + "/" + item))
        for old in imx:
            for young in imy:
                x.append(old)
                y.append(young)
    x = numpy.array(x) / 255
    y = numpy.array(y) / 255
    return x, y


def get_model(inshape=(100, 100, 3), outshape=(100, 100, 3)):
    model = Sequential()
    model.add(Input(shape=inshape))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same"))
    # model.add(AveragePooling2D(pool_size=(2,2)))
    # model.add(Flatten())
    # model.add(Dense(units=10 * 1 * 3, activation="relu"))
    # model.add(Dense(units=10 * 1 * 3, activation="relu"))
    # model.add(Dense(units=100 * 100 * 3, activation="tanh"))
    # model.add(Reshape(target_shape=outshape))
    model.add(Conv2DTranspose(filters=3,kernel_size=(3,3), activation="sigmoid", padding="same"))
    model.compile(loss="mse")
    return model


if __name__ == "__main__":
    x, y = load_images()
    print(x.shape, y.shape)
    model = get_model()
    print(model.summary())
    model.fit(x, y, epochs=1000, steps_per_epoch=10,validation_split=0.2,
              callbacks=[EarlyStopping(patience=5),
                         ModelCheckpoint(filepath=f"models/model",monitor="val_loss", save_best_only=True)])
