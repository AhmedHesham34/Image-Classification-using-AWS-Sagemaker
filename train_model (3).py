#TODO: Import your dependencies.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
import logging
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import smdebug.pytorch as smd
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    model.eval()        
    running_loss=0      
    running_corrects=0  
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)           
        running_corrects += torch.sum(preds == labels.data)     

    total_loss = running_loss // len(test_loader)       
    total_acc = running_corrects.double() // len(test_loader)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}") 
    

def train(model, train_loader, valid_loader, epochs, criterion, optimizer, hook):
    count = 0
    for e in range(epochs):
        print(e)
        model.train()
        hook.set_mode(smd.modes.TRAIN)
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)

        # validation
        model.eval()
        hook.set_mode(smd.modes.EVAL)
        running_corrects=0
        with torch.no_grad():
            for (inputs, labels) in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
        total_acc = running_corrects/ len(valid_loader.dataset)
        logger.info(f"Valid set: Average accuracy: {100*total_acc}%")
        
    return model

def net():
    model = models.resnet18(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    #it should be 133 but i used just 2 because i dont have enought budget to train the 133
    model.fc = nn.Sequential( nn.Linear( num_features, 256),
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 2),
                             nn.ReLU(inplace = True) 
                            )
    return model

def create_data_loaders(data, batch_size):
    
    train_data_path = os.path.join(data, 'train') 
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    #Training data image transformation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ]) 
    #testing data image transformation                                                 
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ]) 
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, validation_data_loader, test_data_loader

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on Device {device}")
    model=net()
    model=model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    train_loader, valid_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    model=train(model, train_loader, valid_loader, args.epochs, loss_criterion, optimizer, hook)
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 256)",
    )
 
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
  
    parser.add_argument('--data', type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    
    main(args)