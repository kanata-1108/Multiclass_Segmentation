import os
from glob import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from make_dataset import MakeDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from multi_unet import Unet
import monai.losses
from monai.networks.utils import one_hot
from monai.metrics.meaniou import MeanIoU

NUM_CLASS = 4
BATCH_SIZE = 8
EPOCHS = 100

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.chdir(os.path.dirname(__file__))

    transform_train = albu.Compose([
        albu.HorizontalFlip(p = 0.5),
        albu.VerticalFlip(p = 0.5),
        albu.Resize(256, 256),
        ToTensorV2()
    ])

    transform_vlaid = albu.Compose([
        albu.Resize(256, 256),
        ToTensorV2()
    ])

    train_dcm, train_msk = sorted(glob('./data_dicom/train/dicom/*.dcm')), sorted(glob('./data_dicom/train/mask/*.png'))
    valid_dcm, valid_msk = sorted(glob('./data_dicom/valid/dicom/*.dcm')), sorted(glob('./data_dicom/valid/mask/*.png'))

    train_dataset = MakeDataset(train_dcm, train_msk, transform = transform_train)
    valid_dataset = MakeDataset(valid_dcm, valid_msk, transform = transform_vlaid)
    print(len(train_dataset))
    print(len(valid_dataset))

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

    model = Unet(in_channels = 1, out_channels = NUM_CLASS).to(device)
    criterion = monai.losses.DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    miou = MeanIoU(reduction = "mean")

    train_loss_value = []
    valid_loss_value = []
    train_iou_value = []
    valid_iou_value = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        valid_loss = 0
        train_iou = 0
        valid_iou = 0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            masks = masks.unsqueeze(1)
            masks = one_hot(masks, num_classes = NUM_CLASS)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = torch.argmax(outputs, dim = 1)
                outputs = outputs.unsqueeze(1)
                outputs = one_hot(outputs, num_classes = NUM_CLASS)
                masks = masks.unsqueeze(1)
                masks = one_hot(masks, num_classes = NUM_CLASS)
                miou(y_pred = outputs, y = masks)

            train_iou = miou.aggregate().item()
            miou.reset()

            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = torch.argmax(outputs, dim = 1)
                outputs = outputs.unsqueeze(1)
                outputs = one_hot(outputs, num_classes = NUM_CLASS)
                masks = masks.unsqueeze(1)
                masks = one_hot(masks, num_classes = NUM_CLASS)
                miou(y_pred = outputs, y = masks)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            valid_iou = miou.aggregate().item()
            miou.reset()
        
        train_loss_value.append(train_loss)
        valid_loss_value.append(valid_loss)
        train_iou_value.append(train_iou)
        valid_iou_value.append(valid_iou)

        print(f'Epoch {epoch + 1} :: train loss {train_loss:.6f}, valid loss {valid_loss:.6f}, train iou {train_iou:.6f}, valid iou {valid_iou:.6f}')
    
    torch.save(model.state_dict(), './dicommodel_weight.pth')

    plt.plot(range(EPOCHS), train_loss_value, c = 'orange', label = 'train loss')
    plt.plot(range(EPOCHS), valid_loss_value, c = 'blue', label = 'valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.savefig(f'./loss.png')
    plt.close()

    plt.plot(range(EPOCHS), train_iou_value, c = 'orange', label = 'train iou')
    plt.plot(range(EPOCHS), valid_iou_value, c = 'blue', label = 'valid iou')
    plt.xlabel('epoch')
    plt.ylabel('iou score')
    plt.grid()
    plt.legend()
    plt.title('iou score')
    plt.savefig(f'./iou.png')