# Deep learning Library
import torch
import torchvision
import torch.utils.data
import json

# Deep learning Model
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data
from tqdm import tqdm
from collections import OrderedDict
from torchvision import models, datasets, transforms

def build_classifier(model):
   
    # Define new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fcl',nn.Linear(25088, 4096,bias=True)),
        ('relu',nn.ReLU(inplace=True)),
        ('drop',nn.Dropout(p=0.)),
        ('fc2',nn.Linear(4096, 1024,bias=True)),
        ('relu2',nn.ReLU(inplace=True)),
        ('drop2',nn.Dropout(p=0.3)),
        ('fc3',nn.Linear(1024, 102,bias=True)),
        ('output',nn.LogSoftmax(dim=1))
    ]))
    
    return classifier

# Set the model to evaluation mode and compute validation loss and accuracy

def train_model(model, trainloader, optimizer, criterion, device):
    
    epochs = 15
    best_acc = 0.0
    
    # Define the scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the classifier
    for epoch in range(epochs):
        train_loss = 0.0
        # Check if a GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Iterate over batches of data
        for i, data in tqdm(enumerate(trainloader)):
            images, labels = data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average training loss and accuracy
        train_loss = train_loss / len(trainloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
    return model

                             
def validation(model, validloader, criterion, device):
    # define criterion
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device),labels.to(device)
            outputs = model.forward(images)
            val_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            val_correct += equality.type(torch.FloatTensor).mean()

    model.train()

    val_loss = val_loss/len(validloader.dataset)
    val_acc = val_correct / len(validloader.dataset)
        
    # Print results
    print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))
    

    
def test_model(model, testloader, device):
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_loss = 0
    test_accuracy = 0
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(testloader)):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            test_accuracy += equality.type(torch.FloatTensor).mean()
            
    model.train()        
    
    print("Test Loss: {:.3f} ".format(test_loss/len(testloader.dataset)),
          "Test Accuracy {:.3f}".format(test_accuracy/len(testloader.dataset)))
    
    
def save_checkpoint(model, save_path, optimizer, train_datasets, epochs, criterion):
    
    # Create a dictionary containing the training dataset
    train_datasets_dict = {'train': train_datasets}

    # Access the class to index mapping for the training dataset
    class_to_idx = train_datasets_dict['train'].class_to_idx
    
    # Save the optimizer state dict
    optimizer_state_dict = optimizer.state_dict()
    
    # TODO: Save the checkpoint 
    checkpoint = {
                  'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'epoch': epochs,
                  'optimizer_state_dict': optimizer_state_dict,
                  'criterion': criterion}

    return torch.save(checkpoint, save_path)
                    
def load_checkpoint(save_path, device, model):
    # create an instance of the VGG16 model
    # Load the pre-trained VGG16 network
    model = models.vgg16(pretrained=True)
    
    # replace the classifier
    model.classifier = build_classifier(model)
    
    # load the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == True:
        checkpoint = torch.load(save_path)
    else:
        checkpoint = torch.load(save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    # create an instance of the optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.01)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
   # retrieve other checkpoint information
    criterion = checkpoint['criterion']
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    
    
    # print a message indicating the checkpoint was loaded
    print(f'Loaded checkpoint from {save_path} (epoch {epoch})')
    
    return model

def predict(processed_image, loaded_model, top_k, gpu):
    
    loaded_model.eval()
    
    # TODO: Implement the code to predict the class from an image file
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processed_image=processed_image.to(gpu)
    loaded_model.to(gpu)
    
    
    with torch.no_grad():
        # Running image through network
        out = loaded_model.forward(processed_image)
        ps = torch.exp(out)
        pbs, indices = torch.topk(ps, top_k)
        pbs = torch.nn.functional.softmax(pbs[0], dim=0)
        pbs = [round(pbs.item()*100, 4) for pbs in pbs]
        
        class_to_idx_inv = {idx: cls for cls, idx in loaded_model.class_to_idx.items()}
        clss = [class_to_idx_inv[idx.item()] for idx in indices[0]]
        
        return pbs, clss