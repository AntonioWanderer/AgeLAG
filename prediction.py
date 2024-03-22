import cv2, os, numpy
from tensorflow.keras.models import load_model

def get_test(path: str="real/"):
    images = []
    names = os.listdir(path)
    for name in names:
        im = cv2.imread(path+name)
        im = cv2.resize(src=im, dsize=(100,100))
        images.append(im)
    images = numpy.array(images)/255
    return images, names

if __name__ == "__main__":
    ims, names = get_test()
    model = load_model("models/model")
    result = (0 + model.predict(ims)) * 255
    for i in range(result.shape[0]):
        cv2.imwrite(filename=f"prediction/{names[i]}", img=result[i])