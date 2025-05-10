import os
from glob import glob
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from make_dataset import MakeDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from multi_unet import Unet
from monai.networks.utils import one_hot
from monai.metrics.meaniou import MeanIoU

NUM_CLASS = 4
BATCH_SIZE = 8

palette = {
    0: (0, 0, 0),    
    1: (200, 0 , 0),  
    2: (0, 200, 0),  
    3: (200, 200, 0)   
}

def overlay_mask(image, mask):

    image = np.squeeze(image)
    image = (image * 255).astype(np.uint8)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(src1 = color_image, alpha = 0.7, src2 = mask, beta = 0.3, gamma = 0)

    return overlay

def calc_iou(pred, mask):

    pred = pred.unsqueeze(1)
    pred = one_hot(pred, num_classes = NUM_CLASS)

    mask = mask.unsqueeze(1)
    mask = one_hot(mask, num_classes = NUM_CLASS)
    miou(y_pred = pred, y = mask)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.chdir(os.path.dirname(__file__))

    transform = albu.Compose([
        albu.Resize(256, 256),
        ToTensorV2()
    ])

    test_dcm, test_msk = sorted(glob('./data_dicom/test/dicom/*.dcm')), sorted(glob('./data_dicom/test/mask/*.png'))
    test_dataset = MakeDataset(test_dcm, test_msk, transform = transform)
    test_loader = DataLoader(test_dataset, shuffle = False, num_workers = 2)
    print(len(test_dataset))

    model = Unet(in_channels = 1, out_channels = NUM_CLASS).to(device)
    model.load_state_dict(torch.load("dicommodel_weight.pth", weights_only = True, map_location = device))

    miou = MeanIoU(reduction = "mean")

    model.eval()
    with torch.no_grad():
        for index, (images, masks) in enumerate(test_loader):
            outputs = model(images.to(device))
            outputs = torch.argmax(outputs, dim = 1)
            calc_iou(pred = outputs, mask = masks.to(device))
            outputs = torch.squeeze(outputs)
            outputs = outputs.to('cpu').detach().numpy()

            height, width = outputs.shape
            color_image = np.zeros((height, width, 3), dtype=np.uint8)

            for class_id, color in palette.items():
                mask = outputs == class_id
                color_image[mask] = color
            
            images = images.to('cpu').detach().numpy()
            images = np.squeeze(images)
            overlay_image = overlay_mask(images, color_image)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(images, cmap='gray')
            # axes[0].axis('off')
            # axes[1].imshow(overlay_image)
            # axes[1].axis('off')
            # plt.savefig('./result/compare_img/compare_'+ str(index) + '.png', bbox_inches='tight')
        
        iou_score = miou.aggregate().item()
        miou.reset()
        print(iou_score)