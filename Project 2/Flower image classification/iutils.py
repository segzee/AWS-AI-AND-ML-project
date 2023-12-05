# Deep learning Library
import torch
import torchvision
import torch.utils.data
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import json

# Deep learning Model
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torch.utils.data

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Call get_data_loaders function with arguments
def get_data_loaders(data_dir):
    train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(30),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    vaild_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    train_datasets = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = torchvision.datasets.ImageFolder(valid_dir, transform=vaild_transforms)
    test_datasets = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=False)
    
    return trainloader, testloader, validloader, train_datasets, test_datasets, valid_datasets

# Function to load and preprocess test image
def  process_image(image_path):
    # Load image from file using Pillow library
    pil_image = Image.open(image_path)

    # Define image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply transformation pipeline to image
    transformed_image = transform(pil_image)

    # Convert transformed image to a PyTorch tensor with batch dimension
    tensor_image = torch.unsqueeze(transformed_image, 0)

    return tensor_image
