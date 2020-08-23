import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from random import randint
from PIL import Image, ImageDraw
import os
import csv
import random
import pickle
from models import *

DATA_SIZE = 224
class ReshapeMe(torch.nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = shape
  def forward(self, X):
    return X.view(self.shape)

class ResNetTransform:
  def __init__(self):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L201
    self.net = torch.load('resnet.pt', map_location=device)

  def __call__(self, tensor):
    with torch.no_grad():
      return self.net(tensor.to(device).unsqueeze(0)).squeeze(0)

class SquarePad:
  pad_color = [0, 0, 0]
  def __init__(self, image_size):
    self.image_size = image_size
    arr = [[self.pad_color for j in range(image_size)] for i in range(image_size)]
    np_arr = numpy.array(arr, dtype=numpy.uint8)
    self.bg = Image.fromarray(np_arr)

  def __call__(self, img):
    width, height = img.size
    scale_f = max(width, height) / self.image_size
    width_new = int(width / scale_f)
    height_new = int(height / scale_f)
    img = img.resize((width_new, height_new))
    copy = self.bg.copy()
    copy.paste(img)
    return copy

device = torch.device("cpu")
model = torch.load("model.pt", map_location=device)
resnet_transform = transforms.Compose([
    transforms.Resize(DATA_SIZE),
    SquarePad(DATA_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ResNetTransform()
])

timeclasses = ['0000_1300', '1301_1400', '1401_1500', '1501_1600', '1601_1700', '1701_1751', '1751_1800', '1801_1850', '1851_1900', '1901_1925', '1926_1950', '1951_1975', '1976_3000']

def year2century(year):
    year = int(year)
    if year % 100 == 0:
        return year // 100
    return year // 100 + 1

def classname2beaty(classname):
    start, end = classname.split('_')
    return f"{start} - {end}"
    result = f"{roman.toRoman(start)} - {roman.toRoman(end)}"
    print(classname, result)
    return result

def predict_date(img):
    print('!!!')
    transformed = resnet_transform(img).unsqueeze(0)
    print('-\n-\n-\n')
    print(transformed.shape)
    predictions = model(transformed).squeeze(0)
    top5 = []
    for i, el in enumerate(list(predictions)):
        top5.append([el.item(), timeclasses[i]])
    top5.sort(reverse=True)
    top5 = top5[:5]
    return [{'date': classname2beaty(el[1]), 'conf': el[0]} for el in top5]
