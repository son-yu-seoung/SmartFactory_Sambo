#globalPath에 현제 실행시킬 환경에 데이터가 저장되어 있는 위치를 입력


from PIL import Image
import os
from glob import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout


class dataSet():
    
    #초기함수
    def __init__(self, globalPath=None):
        self.globalPath = globalPath
        self.x = []
        self.y = []

        #in splitData
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
    
    #이미지를 불러오는 함수 (path = 지금 현재 데이터가 저장되어 있는 파일 주소)
    def imageRead(self, path):
        x = Image.open(path)
        y = path.split("_")[2]

        return x, float(y)

    #실제로 모든 데이터를 읽어들이는 함수
    def getFilesInFolder(self, path):
        #모든 경로들을 다 가져와서 result에 넣음
        train_data = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]
        # print(result)

        for localPath in train_data:
            for i in range(3):
                img, target = self.imageRead(localPath)
                self.x.append(img)
                self.y.append(target)

    #이전에 설정한 dim값으로 데이터 전체를 사이즈를 변환함
    def resizeAll(self, X, Y, dim):
        
        resizedX = []
        resizedY = []

        N = len(X)

        for i in range(N):
            resized = X[i].resize((dim, dim))
            npImg = np.array(resized)

            if len(npImg.shape) == 3:
                resizedX.append(npImg)
                resizedY.append(Y[i])
        
        self.x = np.array(resizedX)
        self.y = np.array(resizedY)
        self.y = np.reshape(self.y, (-1, 1))
    
    #학습데이터랑 테스트데이터로 나누는 함수
    def splitDataset(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2, shuffle=True, stratify=self.y)
        self.test_label = self.test_y

    def onehotencoding(self):
        tempS = set()
        tempKey = list()
        tempVal = list()
        self.tempD = dict()

        for i in range(len(self.train_y)): # PH 종류 받기
            tempS.add(float(self.train_y[i]))
        print('분류해야할 PH 종류개수 :', len(tempS))
        
        tempKey = list(tempS) # srot함수를 사용하기위해 list로 자료형 변환
        tempKey.sort()
        
        for i in range(len(tempKey)): # dict에 key와 value를 넣기위해 리스트 생성
            tempVal.append(i)

        self.tempD = dict(zip(tempKey, tempVal)) # 두 리스트를 이용해 dict 생성 
        
        shape = (len(self.train_y), len(tempS))
        self.train_Y = np.zeros(shape, dtype=int)
        self.test_Y = np.zeros(shape, dtype=int)

        for idx, ph in enumerate(self.train_y):
            onehot_encoding = np.zeros(len(tempS), dtype=int)
            a = self.tempD[float(ph)]
            onehot_encoding[int(a)] = 1
            self.train_Y[idx] = onehot_encoding
        
        for idx, ph in enumerate(self.test_y):
            onehot_encoding = np.zeros(len(tempS), dtype=int)
            a = self.tempD[float(ph)]
            onehot_encoding[int(a)] = 1
            self.test_Y[idx] = onehot_encoding

        with open('../tempD.pkl', 'wb') as fw:
            pickle.dump(self.tempD, fw)
 
    #정규화 함수
    def normZT(self, x):
        x = (x - np.mean(x) / np.std(x))
        return x
    
    #MinMax정규화 함수
    def normMinMax(self, x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    #위에 함수들을 다 실행시키는 함수
    def load_data(self, dim):
        self.getFilesInFolder(self.globalPath) #전체 데이터 가져옴
        self.resizeAll(self.x, self.y, dim) # numpy화 되어 있음
        self.splitDataset() #훈련용, 시험용으로 쪼개기
        self.onehotencoding()
        self.test_X = self.test_x
        self.train_x = self.normZT(self.train_x) #train 정규화
        self.test_x = self.normZT(self.test_x) #test 정규화
        
        return self.train_x, self.train_Y, self.test_x, self.test_Y, self.test_X, self.test_label

class LeNet_train:

    def __init__(self, train_x, train_y, test_x, test_y, test_X, test_label):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.train_X = test_X
        self.test_label = test_label

    def Model(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape = (128,128,3), kernel_size = (5,5), strides = (1,1), filters = 1, padding = "same", activation = "tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters = 6, kernel_size = (5,5), strides = (1,1), padding = "same", activation = "tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters = 16, kernel_size = (5,5), strides = (1,1), padding = "same", activation = "tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate = 0.5),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 256, activation= 'tanh'),
        tf.keras.layers.Dropout(rate = 0.5),
        tf.keras.layers.Dense(units = 84, activation= 'tanh'),
        tf.keras.layers.Dense(units = 84, activation= 'softmax'),
        ])

        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

        model.summary()

        return model

    def train(self):
        model = self.Model()

        #위에서 정의한 모델 학습
        history = model.fit(train_x, train_y, epochs=30, validation_split= 0.2)
        model.save('../LeNet_model.h5')

        plt.figure(figsize = (12,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], 'b-', label = 'loss')
        plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
        plt.xlabel('Epochs')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], 'g-', label = 'accuracy')
        plt.plot(history.history['val_accuracy'], 'k--', label = 'val_accuracy')
        plt.xlabel('Epochs')
        plt.ylim(0.7, 1)
        plt.legend()

        plt.show()

        return model

    def model_pred(self, t):
        model = t
        pred = model.predict(self.test_x)

        for i in range(0, 10):
            PH = np.argmax(pred[i])

            for key, value in ds.tempD.items():
                if PH == value:
                    PH = key
                    break
            
            print('\n정답 : {}, 예상 : {}'.format(float(test_label[i]), PH))

            plt.subplot(1, 10, i+1)
            plt.imshow(test_X[i])
            plt.xlabel(PH)
            plt.ylabel(float(test_label[i]))
        plt.show()
        
    def train_start(self):
        t = self.train()
        self.model_pred(t)
        
globalPath = './train_crop'
ds = dataSet(globalPath)
train_x, train_y, test_x, test_y, test_X, test_label = ds.load_data(128)


LeNet_train = LeNet_train(train_x, train_y, test_x, test_y, test_X, test_label)
LeNet_train.train_start()