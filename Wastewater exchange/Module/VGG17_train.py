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


class dataSet():
    
    #초기함수
    def __init__(self, globalPath):
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
        print(y)
        return x, float(y)

    #실제로 모든 데이터를 읽어들이는 함수
    def getFilesInFolder(self, path):
        #모든 경로들을 다 가져와서 result에 넣음
        train_data = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]
        # print(result)
        print(len(self.y), len(self.x))

        for localPath in train_data:
            img, target = self.imageRead(localPath)
            self.x.append(img)
            self.y.append(target)
    
        return self.x, self.y



    
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
           # print(npImg.shape)
        
        self.x = np.array(resizedX)
        self.y = np.array(resizedY)
        self.y = np.reshape(self.y, (-1, 1))
        #print(self.x.shape, self.y.shape)
    
    #학습데이터랑 테스트데이터로 나누는 함수
    def splitDataset(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2, shuffle=True, stratify=self.y)
        print(self.test_y)
        return self.train_x, self.train_y, self.test_x, self.test_y

    def onehotencoding(self):
        print(self.train_y)
        shape = (len(self.train_y), 12)
        self.train_Y = np.zeros(shape, dtype=int)
        for j in range(len(self.train_y)):
            if self.train_y[j] == 7.78:
                self.train_Y[j] = [1,0,0,0,0,0,0,0,0,0,0,0]
            elif self.train_y[j]  == 7.59:
                self.train_Y[j] = [0,1,0,0,0,0,0,0,0,0,0,0]
            elif self.train_y[j]  == 7.54:
                self.train_Y[j] = [0,0,1,0,0,0,0,0,0,0,0,0]
            elif self.train_y[j]  == 7.49:
                self.train_Y[j] = [0,0,0,1,0,0,0,0,0,0,0,0]
            elif self.train_y[j]  == 7.45:
                self.train_Y[j] = [0,0,0,0,1,0,0,0,0,0,0,0]
            elif self.train_y[j]  == 7.41:
                self.train_Y[j] = [0,0,0,0,0,1,0,0,0,0,0,0]
            elif self.train_y[j]  == 7.35:
                self.train_Y[j] = [0,0,0,0,0,0,1,0,0,0,0,0]
            elif self.train_y[j]  == 7.32:
                self.train_Y[j] = [0,0,0,0,0,0,0,1,0,0,0,0]
            elif self.train_y[j]  == 7.29:
                self.train_Y[j] = [0,0,0,0,0,0,0,0,1,0,0,0]
            elif self.train_y[j]  == 7.28:
                self.train_Y[j] = [0,0,0,0,0,0,0,0,0,1,0,0]
            elif self.train_y[j]  == 7.26:
                self.train_Y[j] = [0,0,0,0,0,0,0,0,0,0,1,0]
            elif self.train_y[j]  == 7.24:
                self.train_Y[j] = [0,0,0,0,0,0,0,0,0,0,0,1]

        shape = (len(self.test_y),12)
        self.test_Y = np.zeros(shape, dtype=int)
        for j in range(len(self.test_y)):
            if self.test_y[j] == 7.78:
                self.test_Y[j] = [1,0,0,0,0,0,0,0,0,0,0,0]
            elif self.test_y[j]  == 7.59:
                self.test_Y[j] = [0,1,0,0,0,0,0,0,0,0,0,0]
            elif self.test_y[j]  == 7.54:
                self.test_Y[j] = [0,0,1,0,0,0,0,0,0,0,0,0]
            elif self.test_y[j]  == 7.49:
                self.test_Y[j] = [0,0,0,1,0,0,0,0,0,0,0,0]
            elif self.test_y[j]  == 7.45:
                self.test_Y[j] = [0,0,0,0,1,0,0,0,0,0,0,0]
            elif self.test_y[j]  == 7.41:
                self.test_Y[j] = [0,0,0,0,0,1,0,0,0,0,0,0]
            elif self.test_y[j]  == 7.35:
                self.test_Y[j] = [0,0,0,0,0,0,1,0,0,0,0,0]
            elif self.test_y[j]  == 7.32:
                self.test_Y[j] = [0,0,0,0,0,0,0,1,0,0,0,0]
            elif self.test_y[j]  == 7.29:
                self.test_Y[j] = [0,0,0,0,0,0,0,0,1,0,0,0]
            elif self.test_y[j]  == 7.28:
                self.test_Y[j] = [0,0,0,0,0,0,0,0,0,1,0,0]
            elif self.test_y[j]  == 7.26:
                self.test_Y[j] = [0,0,0,0,0,0,0,0,0,0,1,0]
            elif self.test_y[j]  == 7.24:
                self.test_Y[j] = [0,0,0,0,0,0,0,0,0,0,0,1]

        for i in range (1,len(self.test_Y)):
            print(self.test_Y[i])

        
        return self.train_Y, self.test_Y

    def shuffleData(self):
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        self.test_x = np.array(self.train_y)
        self.test_y = np.array(self.test_y)
        self.train_x, self.train_y = shuffle(self.train_x, self.train_y, self.test_x, self.test_y)
        print(self.train_y)
        return self.train_x, self.train_y, self.test_x, self.test_y
    

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
        

        return self.train_x, self.train_Y, self.test_x, self.test_Y, self.test_X



globalPath = './crop_img'
ds = dataSet(globalPath)
train_x, train_y, test_x, test_y, test_X = ds.load_data(128)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape = (128,128,3),kernel_size = (3,3), filters = 32, padding = 'same', activation = 'relu'),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 64, padding = 'same', activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size = (2,2)),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 128, padding = 'same', activation = 'relu'),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 256, padding = 'valid', activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size = (2,2)),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units = 512, activation= 'relu'),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Dense(units = 256, activation= 'relu'),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Dense(units = 12, activation= 'softmax'),
])

model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

#위에서 정의한 모델 학습
history = model.fit(train_x, train_y, epochs=1000, validation_split= 0.2)

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


model.evaluate(test_x, test_y, verbose = 0)
pred = model.predict(test_x)

for i in range(1, 10):
    print('정답 :', test_y[i])
    print('예상 :', pred[i])

for i in range(0, 10):
    PH = np.argmax(pred[i])

    if PH == 0:
        PH = '7.78'
    elif PH == 1:
        PH = '7.59'
    elif PH == 2:
        PH = '7.54'
    elif PH == 3:
        PH = '7.49'
    elif PH == 4:
        PH = '7.45'
    elif PH == 5:
        PH = '7.41'
    elif PH == 6:
        PH = '7.35'
    elif PH == 7:
        PH = '7.32'
    elif PH == 8:
        PH = '7.29'
    elif PH == 9:
        PH = '7.28'
    elif PH == 10:
        PH = '7.26'
    elif PH == 11:
        PH = '7.25'

    plt.subplot(1, 10, i+1)
    plt.imshow(test_X[i])
    plt.xlabel(PH)
plt.show()