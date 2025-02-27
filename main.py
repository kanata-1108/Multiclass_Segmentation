import os
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from multi_unet import Unet
from monai.metrics.meaniou import MeanIoU

class MakeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.dir_length = len(os.listdir(image_dir))
        self.image_paths = sorted([f.resolve() for f in Path(image_dir).iterdir() if f.is_file()])
        self.mask_paths = sorted([f.resolve() for f in Path(mask_dir).iterdir() if f.is_file()])
        self.transform = transform

    def __getitem__(self, index):
        
        # 入力画像の読み込み
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # マスクデータの読み込み
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        # 前処理が指定されていれば前処理を実行
        if self.transform is not None:
            transformd = self.transform(image = img, mask = mask)
            dicom_array, mask = transformd['image'], transformd['mask']

        # albuのToTensorはtorchvisionのToTensorのように規格化してくれないらしい
        dicom_array, mask = dicom_array / 255., mask / 255.

        # マスクの方にバッチ用の次元がないので追加
        mask = mask.unsqueeze(0)

        return dicom_array.to(torch.float32), mask.to(torch.float32)
    
    def __len__(self):
        return self.dir_length
    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.chdir(os.path.dirname(__file__))

    train_path = "./data/train/"
    valid_path = "./data/valid/"
    test_path = "./data/test/"

    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(p = 0.5),
        # transforms.RandomVerticalFlip(p = 0.5),
        # transforms.RandomInvert(p = 0.5),
        transforms.Resize(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = MakeDataset(train_path + 'train_org', train_path + 'train_annotated', transform = ToTensorV2())
    valid_dataset = MakeDataset(valid_path + 'valid_org', valid_path + 'valid_annotated', transform = ToTensorV2())
    test_dataset = MakeDataset(test_path + 'test_org', test_path + 'test_annotated', transform = ToTensorV2())

    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = False)
    test_loader = DataLoader(test_dataset, num_workers = 2, shuffle = False)

    img, mask = train_dataset[0]
    unique_value, _ = torch.unique(mask, return_counts = True)

    epochs = 100
    model = Unet(in_channels = 3, out_channels = unique_value)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    mean_iou = MeanIoU()

    for epoch in range(epochs):
        train_loss = 0

        for (inputs, masks) in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for (inputs, masks) in train_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                mean_iou(y_pred = outputs, y = masks)
            train_iou = mean_iou.aggregate().item()
            mean_iou.reset()

        print(f'Epoch {epoch + 1} :: train loss {train_loss:.6f}, train iou {train_iou:.6f}')            