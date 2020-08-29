import torch
import numpy
import pickle
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from random import randint
from PIL import Image, ImageDraw
from models import ReshapeMe

device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda:0')

class PassLayer(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, X):
    return X

class ResNetTransform:
  def __init__(self):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L201
    model_ft = models.vgg19(pretrained=True)
    self.net = torch.nn.Sequential(
       *model_ft.features[:7],
    ).to(device)
    print(self.net)


  def __call__(self, tensor):
    with torch.no_grad():
      return self.net(tensor.to(device).unsqueeze(0)).cpu()

DATA_SIZE = 56

class MaskTransform:
  fill_color = (0, 0, 0)
  def __init__(self, mask_size=28):
    assert(DATA_SIZE % mask_size == 0)
    self.mask_size = mask_size

  def __call__(self, img):
    k =  (DATA_SIZE // self.mask_size)
    n = randint(0, k**2 - 1)
    row = n // k
    col = n % k
    x1 = row * self.mask_size
    y1 = col * self.mask_size
    x2 = x1 + self.mask_size
    y2 = y1 + self.mask_size
    for i in range(x1, x2):
      for j in range(y1, y2):
        img.putpixel((i, j), self.fill_color)
        # img[i][j] = self.fill_color
    return img

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
    return img
    copy = self.bg.copy()
    copy.paste(img)
    return copy

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class MyTransform:
    def __init__(self):
        self.call = transforms.Compose([
            SquarePad(DATA_SIZE * 10),
            transforms.CenterCrop(DATA_SIZE),
            transforms.ColorJitter(0.1), # Randomly change the brightness, contrast and saturation of an image.
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ResNetTransform(),
            gram_matrix,
            ReshapeMe(-1)
        ])
    def __call__(self, img):
        return self.call(img)

data_transforms = {
    'train': MyTransform(),
    'val': MyTransform(),
}

if __name__ == "__main__":
    device = torch.device('cpu')
    resnet_transform = ResNetTransform()
    transform = transforms.Compose([
        transforms.Resize(DATA_SIZE),
        SquarePad(DATA_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        resnet_transform
    ])
    torch.save(resnet_transform.net.cpu(), 'resnet.pt')
