# Use Segmentation Models Pytorch

import os
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from multi_unet import Unet
import monai.losses
from monai.networks.utils import one_hot
from monai.metrics.meaniou import MeanIoU

CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
BATCH_SIZE = 8
EPOCHS = 50

class MakeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):

        self.dir_length = len(os.listdir(image_dir))
        self.image_paths = sorted([f.resolve() for f in Path(image_dir).iterdir() if f.is_file()])
        self.mask_paths = sorted([f.resolve() for f in Path(mask_dir).iterdir() if f.is_file()])
        self.transform = transform

    def __getitem__(self, index):

        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformd = self.transform(image = img, mask = mask)
            image, mask = transformd['image'], transformd['mask']

        mask = mask.to(torch.long)
        return image, mask
    
    def __len__(self):
        return self.dir_length

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.chdir(os.path.dirname(__file__))

    transform_train = albu.Compose([
        albu.Resize(256, 256),
        albu.Normalize(mean = (0, 0, 0), std = (1, 1, 1)),
        ToTensorV2()
    ])

    train_dir = "./data/train/"

    train_dataset = MakeDataset(train_dir + "train_org", train_dir + "train_annotated", transform = transform_train)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    model = Unet(in_channels = 3, out_channels = len(CLASSES)).to(device)
    criterion = monai.losses.DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    miou = MeanIoU()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            masks = masks.unsqueeze(1) # outputsと形をそろえるために次元を追加
            masks = one_hot(masks, num_classes = len(CLASSES)) # (batch_size, 1, 256, 256) -> (batch_size, 12, 256, 256)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                masks = masks.unsqueeze(1)
                masks = one_hot(masks, num_classes = len(CLASSES))
                miou(y_pred = outputs, y = masks)
            train_iou = miou.aggregate().item()
            miou.reset()

        print(f"Epoch {epoch + 1} / {EPOCHS}:: Loss: {running_loss / len(train_loader)}, IoU: {train_iou}")