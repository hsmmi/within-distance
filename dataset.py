import numpy as np

from my_io import read_dataset_to_X_and_y


class Dataset():
    def __init__(self, file: str):
        '''
        Just create all the needed variables
        '''
        self.sample, self.label = read_dataset_to_X_and_y(file)
        self.number_of_sample = self.sample.shape[0]
        self.number_of_feature = self.sample.shape[1]
        self.representor = np.zeros_like(self.sample[0])
