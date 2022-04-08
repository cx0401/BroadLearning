import scipy.io as scio
import numpy as np


class matData():

    def loadmat(self, dataFile):
        data = scio.loadmat(dataFile)
        self.train_x = np.double(data['train_x'] / 255)
        self.train_y = np.double(data['train_y'])
        self.test_x = np.double(data['test_x'] / 255)
        self.test_y = np.double(data['test_y'])
        return

    def save(self, output, data):
        return output.write(data)
