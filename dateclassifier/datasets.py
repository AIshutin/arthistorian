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
import json
from tqdm.notebook import tqdm

device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda:0')

timeclasses = json.load(open('timeclasses.json'))

def find_class_id(year):
  original_year = year
  year = year.replace('c.', '')
  lbl = ''
  for el in year:
    if el not in '0123456789.':
        continue
    lbl = lbl + el
  try:
      year = int(float(lbl))
  except:
      print(year)
      return None
  for i in range(len(timeclasses)):
    start, end = timeclasses[i].split('_')
    start = int(start)
    end = int(end)
    if start <= year <= end:
      if i == 0:
          logging.debug(f"{original_year} -> {year}")
      return i
  logging.warning(f"{year} is not in classification")
  return None

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
  cnts = [0] * len(timeclasses)

  def __init__(self, path="../data/artdataset", transforms=data_transforms['train']):
    self.path = path
    self.transforms = transforms
    if path is None:
        return
    self.data = []
    files = set(os.listdir(os.path.join(path, 'art')))
    with open(os.path.join(path, 'info.csv'), newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      header = True
      for row in spamreader:
        if header:
          header = False
          continue
        path, label = ','.join(row).split(',')
        if path not in files:
          continue
        # print(labels)
        self.data.append((path, int(float(label))))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    img = Image.open(os.path.join(self.path, 'art', self.data[i][0]))
    return self.transforms(img), int(self.data[i][1])

class FeatureDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        cnt = 0
        for el in tqdm(dataset, desc='processing data'):
            self.data.append((el[0].cpu(), el[1]))
            cnt += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

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
