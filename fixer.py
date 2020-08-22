from flask import jsonify
from PIL import Image
import PIL
import os
import sys
from tqdm import tqdm
import logging

def resize_img(img, size=224):
    width, height = img.size
    scale_f = max(width, height) / size
    width_new = int(width / scale_f)
    height_new = int(height / scale_f)
    img = img.resize((width_new, height_new))
    return img

source = sys.argv[1]
dest = sys.argv[2]

for el in tqdm(os.listdir(source)):
    try:
        img = Image.open(os.path.join(source, el))
        img = resize_img(img)
        img.convert("RGB").save(os.path.join(dest, el))
    except OSError as exp:
        logging.warning(exp)
        logging.warning(el)
    except PIL.Image.DecompressionBombError:
        pass
