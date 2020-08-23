import torch
import numpy
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from random import randint
from PIL import Image, ImageDraw
import os
import csv
import random
from tqdm.notebook import tqdm

class NormalizeTensor(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, X):
    return torch.nn.functional.normalize(X, dim=1)

class Flatten(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, X):
    return torch.flatten(X, 1)

class PrintMe(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, X):
    print('shape', X.shape, X.device)
    return X

class ReshapeMe(torch.nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = shape
  def forward(self, X):
    return X.view(self.shape)

def create_model(N=512, embeddings=True):
  model_ft = models.resnet18(pretrained=True)
  modules = [
      model_ft.layer4,
      model_ft.avgpool,
      ReshapeMe([-1, 512]),
      torch.nn.Linear(512, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, N),
    ]
  if embeddings:
    modules.append(NormalizeTensor())
  else:
    modules.append(torch.nn.Softmax())
  net = torch.nn.Sequential(*modules)
  return net.train()

if __name__ == "__main__":
    NormalizeTensor()(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
