import os
import numpy as np

from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

class DataProcessing:
    def __init__(self, path):
        self.path = path

        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
    
    def getFilesInFolder(self, path):
        train_path = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))]

        for local_path in train_path: # read_image and target
            self.train_x.append(Image.open(local_path))
            self.train_y.append(float(local_path.split("_")[2])) # '7.24' -> 7.24
    
    def resize_img(self, dim):

        for i in range(len(self.train_x)):
            self.train_x[i] = np.array(self.train_x[i].resize((dim, dim)))
        
        # list -> numpy 
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)

    def split_dataset(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.train_x, self.train_y, test_size=0.3, random_state=777)

    def normalize_max(self, y, maxVal):
        self.train_y = y / maxVal

    def min_max(self, y):
        self.train_y = (y - np.min(y)) / (np.max(y) - np.min(y))

    def load_data(self, dim):
        self.getFilesInFolder(self.path)
        self.resize_img(dim)
        #self.min_max(self.train_y)
        self.normalize_max(self.train_y, 10)
        self.split_dataset()

        return self.train_x, self.train_y, self.test_x, self.test_y



