import os

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class MyDataset(Dataset):
    def __init__(self,
                 df,
                 transform=None
                 ):
        self.df = df.copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __get_input(self, row):

        image_a = cv2.imread(row['image_a'])
        image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)

        image_b = cv2.imread(row['image_b'])
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)

        inputs = {'first': image_a, 'second': image_b}

        return inputs

    def __get_output(self, row):
        label = row['label']
        return label

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        inputs = self.__get_input(row)
        labels = self.__get_output(row)

        if self.transform:
            image_a = inputs['first']
            image_b = inputs['second']

            transformer = self.transform(image=image_a, image_b=image_b)

            image_a = transformer['image']
            image_b = transformer['image_b']

            inputs = {'first': image_a, 'second': image_b}

        return index, inputs, labels