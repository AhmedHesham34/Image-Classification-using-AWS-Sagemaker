#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import time
import os
import sys
import logging

from PIL import ImageFile
#some image didn't ploaded well
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    Accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            loss=criterion(output, target)
            _, predictions = torch.max(output, 1)
            test_loss += loss.item() * data.size(0)
            Accuracy += torch.sum(predictions == target.data)

        total_loss = test_loss // len(test_loader)
        Accuracy = Accuracy // len(test_loader)
        print(f'Test set: Accuracy: {Accuracy}/{len(test_loader.dataset)} = {100*Accuracy}%),\t Testing Loss: {total_loss}')


def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    EDIT: Complete this function that can take a model and
          data loaders for training and will get train the model
    '''

    image_dataset={'train':train_loader}
    
    for epoch in range(epochs):
        
            model.train()
            los = 0
            acc = 0

            for inputs, datalabels in image_dataset["train"]:
                inputs=inputs.to(device)
                datalabels=datalabels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, datalabels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, prediction = torch.max(outputs, 1)
                los += loss.item() * inputs.size(0)
                acc += torch.sum(prediction == datalabels.data)
            epoch_loss = los // len(image_dataset["train"])
            epoch_acc = acc // len(image_dataset["train"])
            logger.info("Epoch {}, loss: {}, acc: {}\n".format(epoch, epoch_loss, epoch_acc))           
    return model
    
def net():

    model = models.resnet18(pretrained=True) 

    for param in model.parameters():
        param.requires_grad = False 
        nfeatures = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(nfeatures, 512),
                   nn.ReLU(inplace = True),
                   nn.Linear(512, 256), 
                   nn.ReLU(inplace=True),
                   nn.Linear(256, 2)) # it should be 133 but i only used 2 because i dont have enought budget to train 133.

     
    return model

def create_data_loaders(data, batch_size):
    
    train_data_path = os.path.join(data, 'train') # Calling OS Environment variable and split it into 3 sets
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ]) # transforming the training image data
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ]) # transforming the testing image data

    # loading train,test & validation data from S3 location using torchvision datasets' Imagefolder function
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_loader, test_loader, validation_loader

def main(args):
    '''
    EDIT: Initialize a model by calling the net function
    '''
    device =   torch.device("cpu")
    model=net()
    model=model.to(device)
 
    '''
    EDIT: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    EDIT: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    logger.info(int(args.batch_size))
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, int(args.batch_size))
    
    logger.info("Train Model")
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    EDIT: Test the model to see its accuracy
    '''
    logger.info("Test the Model")
    
    test(model, test_loader, loss_criterion, device)
    
    '''
    EDIT: Save the trained model
    '''
    logger.info("Save the  Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    EDIT: Specify any training args that you might need
    '''
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        metavar="LR", 
        help="learning rate"
    )
   
    
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args=parser.parse_args()
    
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    
    main(args)