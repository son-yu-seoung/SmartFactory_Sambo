from PIL import Image
import os
import numpy as np
from glob import glob
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

class LeNet_predict:
    def __init__(self, path):
        self.new_model = tf.keras.models.load_model('./LeNet_model.h5')

        with open('./tempD.pkl', 'rb') as fr:
            self.tempD = pickle.load(fr)
        self.image = []

        self.path = path

    
    def pred_start(self):
        print(self.image.shape) # (512, 512) -> (1, 512, 512)

        pred = self.new_model.predict(self.image)
        PH = []
        
        for i in range(len(pred)):
            max_idx = np.argmax(pred[i])
            for key, value in self.tempD.items():
                if max_idx == value:
                    PH.append(key)
                    break
            # print('{}PH'.format(PH[i]))

        return PH
    
    def image_Read(self, path):
        image = Image.open(path)

        return image

    def getFilesInFolder(self, path):
        #모든 경로들을 다 가져와서 result에 넣음
        image_path = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]


        for localPath in image_path:
            img = self.image_Read(localPath)
            self.image.append(img)

        return self.image
    
    def resizeAll(self, X, dim):
        resized_image = []

        N = len(X)

        for i in range(N):
            resize_img = X[i].resize((dim, dim))
            resize_np = np.array(resize_img)
            resized_image.append(resize_np)

        self.image = np.array(resized_image)

    def predict(self, dim):
        self.getFilesInFolder(self.path)
        self.resizeAll(self.image, dim)

        return self.pred_start()

# path = './train_crop/'
# LeNet = LeNet_predict(path)
# LeNet.predict(128)