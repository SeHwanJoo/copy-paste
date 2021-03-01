import os
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt

def check(i, lst):
    return i not in lst

class CustomDataset:
    def __init__(self, df, fold, CFG, normal=True):
        self.df = df
        self.image_ids = fold['image_id'].unique()
        self.CFG = CFG
        self.normal = normal
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.df.shape[0]:
            raise StopIteration
        while self.df.loc[self.index, 'image_id'] not in self.image_ids:
            self.index += 1
        if self.CFG.image_size == 'original':
            image_path = os.path.join(self.CFG.image_path, 'train', self.df.loc[self.index, 'image_id'] + '.jpg')
        else:
            image_path = os.path.join(self.CFG.image_path, 'train', self.df.loc[self.index, 'image_id'] + '.png')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box = [self.df.loc[self.index, 'x_min_resized'], self.df.loc[self.index, 'y_min_resized'],
               self.df.loc[self.index, 'x_max_resized'], self.df.loc[self.index, 'y_max_resized'],
               self.df.loc[self.index, 'class_id']]
        original_box = [self.df.loc[self.index, 'x_min'], self.df.loc[self.index, 'y_min'],
               self.df.loc[self.index, 'x_max'], self.df.loc[self.index, 'y_max'],
               self.df.loc[self.index, 'dim0'], self.df.loc[self.index, 'dim1']]
        # if self.normal is not True:
        #     x_min = self.df[self.df['image_id'] == self.image_ids[self.index]]['x_min_resized'].values.tolist()
        #     x_max = self.df[self.df['image_id'] == self.image_ids[self.index]]['x_max_resized'].values.tolist()
        #     y_min = self.df[self.df['image_id'] == self.image_ids[self.index]]['y_min_resized'].values.tolist()
        #     y_max = self.df[self.df['image_id'] == self.image_ids[self.index]]['y_max_resized'].values.tolist()
        #     class_ids = self.df[self.df['image_id'] == self.image_ids[self.index]]['class_id'].values.tolist()
        # for i in range(len(x_min)):
        #     box.append([x_min[i], y_min[i], x_max[i], y_max[i], class_ids[i]])
        self.index += 1
        return img, box, original_box

    def __len__(self):
        return self.df.shape[0]


class NormalDataset():
    def __init__(self, df, CFG, normal=True):
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.CFG = CFG
        self.normal = normal
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.CFG.image_size == 'original':
            image_path = os.path.join(self.CFG.image_path, 'train', self.df.loc[self.index, 'image_id'] + '.jpg')
        else:
            image_path = os.path.join(self.CFG.image_path, 'train', self.df.loc[self.index, 'image_id'] + '.png')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_id = self.image_ids[self.index]
        self.index += 1
        return img, image_id

    def __len__(self):
        return self.df.shape[0]
