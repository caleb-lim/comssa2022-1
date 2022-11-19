import os
from keras import models
import cv2
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_DIR = "C://Users//19754532//Downloads//"
print(os.listdir(MODEL_DIR))
model = models.load_model(MODEL_DIR + 'signature_forgery_one_shot.h5')

def check_forgery(path_img_1, path_img_2):
    img1 = cv2.imread(path_img_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path_img_2, cv2.IMREAD_GRAYSCALE)
    img1 = img1.reshape((1, 268, 650, 1))
    img2 = img2.reshape((1, 268, 650, 1))
    img1 = img1.astype('float32') / 255
    img2 = img2.astype('float32') / 255

    if model.predict((img1, img2))[0][0] >= 0.5:
        return 'Genuine Signatures'
    else:
        return 'Forged Signatures'

print(check_forgery(MODEL_DIR + 'test_data/063/01_063.png', MODEL_DIR + 'test_data/063_forg/01_0104063.PNG'))
