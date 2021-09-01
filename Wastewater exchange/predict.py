from PIL import Image
import os
import numpy as np
from glob import glob
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

class PH_predict:
    def __init__(self, path):
        self.model = tf.keras.models.load_model('./VGG_model.h5')
        self.path = path

        self.realT = []

    def getFilesInFolder(self, path):
        realT_path = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))]

        for local_path in realT_path: # read_image and target
            self.realT.append(Image.open(local_path))
    
    def resize_img(self, dim):

        for i in range(len(self.realT)):
            self.realT[i] = np.array(self.realT[i].resize((dim, dim)))
        
        # list -> numpy 
        self.realT = np.array(self.realT)
    
    def normalize_max(self, maxVal):
        self.realT = self.realT / maxVal
    
    def pred_start(self):
        print(self.realT.shape) # (512, 512) -> (1, 512, 512)

        pred = self.model.predict(self.realT)
        pred_PH = pred * 10

        self.realT = []

        return pred_PH

    def predict(self, dim):
        
        self.getFilesInFolder(self.path)
        self.resize_img(dim)
        self.normalize_max(10)

        return self.pred_start()
