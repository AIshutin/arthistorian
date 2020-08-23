import torch
import numpy
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
from random import randint
from PIL import Image, ImageDraw
from transforms import data_transforms
import logging
import os
import csv
import random

timeclasses = ['0000_1300', '1301_1400', '1401_1500', '1501_1600', '1601_1700', '1701_1751', '1751_1800', '1801_1850', '1851_1900', '1901_1925', '1926_1950', '1951_1975', '1976_3000']
device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda:0')

def find_class_id(year):
  year = int(float(year))
  for i in range(len(timeclasses)):
    start, end = timeclasses[i].split('_')
    start = int(start)
    end = int(end)
    if start <= year <= end:
      return i
  logging.warning(f"{year} is not in classification")
  return len(timeclasses)

class MNISTData(Dataset):
    classes = 10
    def __init__(self, transforms=data_transforms['train']):
        self.data = datasets.MNIST('../data', train=True, download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        img, label = self.data[ind]
        # print(img, label)
        return self.transforms(img), label

class TimeDataset(Dataset):
  classes = len(timeclasses)
  def __init__(self, path="../../../artdataset", key=-1, transforms=data_transforms['train']):
    self.path = path
    self.transforms = transforms
    if key is None:
      return
    self.data = []
    files = set(os.listdir(os.path.join(path, 'art')))
    with open(os.path.join(path, 'data.csv'), newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      header = True
      for row in spamreader:
        if header:
          header = False
          continue
        # print(row)
        path, *labels = ','.join(row).split(',')
        if path not in files:
          continue
        # print(labels)
        lbl = ''
        for el in labels[key]:
          if el not in '0123456789':
            continue
          lbl = lbl + el
        if lbl == '':
          continue
        if 'c.' in labels[key]:
          labels[key] = labels[key][2:]
        self.data.append((path, find_class_id(labels[key])))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    img = Image.open(os.path.join(self.path, 'art', self.data[i][0]))
    return self.transforms(img), int(self.data[i][1])

class MiniDataset(Dataset):
    def __init__(self, dataset, mx_size):
        ind = [i for i in range(len(dataset))]
        random.shuffle(ind)
        self.classes = dataset.classes
        ind = ind[:mx_size]
        self.dataset = dataset
        self.data = ind

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.dataset[self.data[i]]

def split_dataset(dataset, p=0.3):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                [len(dataset) - int(len(dataset) * p), int(len(dataset) * p)])
    test_dataset.transforms = data_transforms['val']
    return train_dataset, test_dataset
