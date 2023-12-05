# Imports here
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

from iutils import get_data_loaders, process_image, test_dir
from function import save_checkpoint, load_checkpoint, predict, test_model


#Define the command-line arguments
parser = argparse.ArgumentParser(description='Image classification prediction')

parser.add_argument('--image_path', type=str,default = test_dir +"/102/image_08004.jpg", help='Path to the image file')

parser.add_argument('-s','--save_path', action='store', default = 'checkpoint.pth', help='Enter location to save checkpoint in.')

parser.add_argument('--arch', action='store', default='vgg16', dest='arch', help='choose architecture')

parser.add_argument('--checkpoint', type=str, help='Path to the saved PyTorch model checkpoint')

parser.add_argument('-k','--top_k', type=int, default=5,  help='Return top K most likely classes')

parser.add_argument('--category_names', action="store_true", help="Use real category names")

parser.add_argument("-d", "--display", action="store_true", help="Display predictions as image")

parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json', help='Enter path to image.')

parser.add_argument('-de','--device', type=str, default='False', help='use GPU if available')


args = parser.parse_args()  


save_path = args.save_path
image_path = args.image_path
top_k = args.top_k
gpu = args.device
cat_names = args.cat_to_name

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Establish model template
pre_model = args.arch
model = getattr(models,pre_model)(pretrained=True)

# Load model
loaded_model = load_checkpoint(save_path, gpu, model)

# Preprocess image - assumes jpeg format
processed_image = process_image(image_path)

# Carry out prediction
probs, classes = predict(processed_image, loaded_model, top_k, gpu)

# Print probabilities and predicted classes
print(probs)
print(classes) 

names_flowers = []
for i in classes:
    names_flowers += [cat_to_name[i]]
    
# Print name of predicted flower with highest probability
print(f"This flower is most likely to be a: '{names_flowers[0]}' with a probability of {round(probs[0]*100,4)}% ")
