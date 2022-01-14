from PIL import Image
import os
import numpy as np
from glob import glob
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import pandas as pd

class PH_predict:
    def __init__(self, path):
        self.model = tf.keras.models.load_model('./VGG_model.h5') #이전에 학습한 VGG모델의 가중치와 여러가지를 들고옴
        self.path = path

        self.realT = [] # 사진이 담길 리스트

    def getFilesInFolder(self, path): # 폴더 안에 있는 파일의 주소를 가져옴
        realT_path = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))]

        for local_path in realT_path: # read_image and target
            self.realT.append(Image.open(local_path))
    
    def resize_img(self, dim): # 찍어온 이미지를 사이즈를 일정하게 바꿈

        for i in range(len(self.realT)):
            self.realT[i] = np.array(self.realT[i].resize((dim, dim)))
        
        # list -> numpy 
        self.realT = np.array(self.realT)
    
    def normalize_max(self, maxVal): #Max정규화를 하는 함수
        self.realT = self.realT / maxVal
    
    def pred_start(self): # 실제로 예측을 하는 함수
        # print(self.realT.shape) # (512, 512) -> (1, 512, 512)

        pred = self.model.predict(self.realT) #예측을 하고 예측한 데이터를 pred에 저장
        pred_PH = pred * 10 # pred에 10을 곱해서 수치를 맞춘다.

        self.realT = []
        
        print(pred_PH.shape)
        #B8~11
        PH = pd.DataFrame([0,0,0,0,0,0],index=['A', 'B', 'C', 'D', 'E', 'F'], columns = ["PH"])
        real_PH = pd.DataFrame(pred_PH, index=['1', '2', '3', '4', '5', '6', '7'], columns=['PH'])
        PH = PH.append(real_PH)
        PH.drop(['A', 'B', 'C', 'D', 'E', 'F'])
        print(PH)
        PH.to_excel('./PH.xlsx', sheet_name='new_name')
        
        return pred_PH

    def predict(self, dim): # 위에 있는 함수를 시작함
        
        self.getFilesInFolder(self.path)
        self.resize_img(dim)
        self.normalize_max(10)

        return self.pred_start()


A = PH_predict("./realT_crop/")
B = A.predict(256)

print('#################')
print(B)
