import dataProcessing as dp
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split

def Model(inputs=(256, 256, 3)):

    model = tf.keras.Sequential([
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputs),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)), # 64
        
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)), # 32

        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)), # 16

        Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)), # 8

        Flatten(),
        Dense(1024, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.summary()

    return model

def train_model(input_shape, train_x, train_y, epoch):
    model = Model(input_shape) 

    history = model.fit(train_x, train_y, epochs=epoch, validation_split=0.2)
    model.save('../VGG_model.h5')

    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'b-', label = 'loss')
    plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
    plt.xlabel('Epochs')
    plt.ylim(0, 0.01)
    plt.legend()

    plt.show()

    return model

def pred_model(trained_model, test_x, test_y):
    model = trained_model # 이렇게 model을 새로 받으면 가중치 초기화 아닌가? 테스트하기
    pred = model.predict(test_x)
    test_y = test_y * 10 # normalize 풀기
    PH = pred * 10

    for i in range(10):
        #print('\n정답 : {}, 예상 : {}'.format(test_y[i], PH[i]))
        print(test_y[i], PH[i][0])

        plt.subplot(1, 10, i+1)
        plt.imshow(test_x[i])
        plt.xlabel(PH[i][0])
        plt.ylabel(test_y[i])

path = './train_crop'
dim = 256
pre_p = dp.DataProcessing(path)
train_x, train_y, test_x, test_y = pre_p.load_data(dim)

trained_model = train_model((dim, dim, 3), train_x, train_y, 10)
pred_model(trained_model, test_x, test_y)