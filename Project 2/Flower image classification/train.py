import argparse

# Deep learning Library
import torch
import torchvision
import torch.utils.data
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import json

# Deep learning Model
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from iutils import get_data_loaders
from tqdm import tqdm
from function import build_classifier, validation, train_model, test_model, save_checkpoint

parser = argparse.ArgumentParser(description='Utilize a neural network to perform image prediction.')

# Create argparser
parser.add_argument('-i', '--image_path', action='store', type=str, default='./flowers', help='path to the directory containing the image data')

parser.add_argument('-s', '--save_path', action='store',  default='checkpoint.pth', help='location for saving model checkpoint')

parser.add_argument('--arch', action='store', default='vgg16', dest='arch', help='choose architecture')

parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate for the optimizer')

parser.add_argument('-d', '--dropout', action='store', type=float, default=0.3, metavar='', help='dropout probability for the classifier')

parser.add_argument('-e', '--epoch', action = 'store', type=int, default=15, metavar='', help='number of epochs for training')

parser.add_argument('-de', '--device', action='store_true',help='use GPU if available')


results = parser.parse_args()
                                                           
image_path = results.image_path
save_path = results.save_path
learning_rate = results.learning_rate
epochs = results.epoch
gpu = results.device
                  
# Load and preprocess data 
trainloader, testloader, validloader, train_datasets, test_datasets, valid_datasets = get_data_loaders(image_path)
                    
# Load pretrained model
pre_model = results.arch
model = getattr(models,pre_model)(pretrained=True)

# zero grad
for param in model.parameters():
   param.requires_grad = False

# Attach new classifier to the model
model.classifier = build_classifier(model)

# condition for optimizer
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.01)
# define criterion
criterion = nn.CrossEntropyLoss()
                 
# Train model
model = train_model(model, trainloader, optimizer, criterion, gpu)

# Valid model 
validation(model, validloader, criterion, gpu)
    
# Test model
test_model(model, testloader, gpu)

# Save model
save_checkpoint(model, save_path, optimizer, train_datasets, epochs, criterion)
print('Model checkpoint saved successfully.')
