import cv2
import torch
import pydicom
from torch.utils.data import Dataset

class MakeDataset(Dataset):
    def __init__(self, dicom_list, mask_list, transform):
        self.dir_length = len(dicom_list)
        self.dcm_paths = dicom_list
        self.mask_paths = mask_list
        self.transform = transform

    def __getitem__(self, index):
        
        # DICOMデータの読み込み
        data = pydicom.dcmread(self.dcm_paths[index], force = True)

        # ウィンドウ処理(0~255の値に変換)
        window_center = 30
        window_width = 330
        dicom_array = data.pixel_array

        slope = getattr(data, "RescaleSlope")
        intercept = getattr(data, "RescaleIntercept")

        dicom_array = slope * dicom_array + intercept

        window_max = window_center + window_width / 2
        window_min = window_center - window_width / 2
        dicom_array = 255 * (dicom_array - window_min) / (window_max - window_min)
        dicom_array[dicom_array > 255] = 255
        dicom_array[dicom_array < 0] = 0

        # マスクデータの読み込み
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        # 前処理が指定されていれば前処理を実行
        if self.transform is not None:
            transformd = self.transform(image = dicom_array, mask = mask)
            dicom_array, mask = transformd['image'], transformd['mask']

        # albuのToTensorはtorchvisionのToTensorのように規格化してくれないらしい
        dicom_array= dicom_array / 255.

        mask = mask.to(torch.long)

        return dicom_array.to(torch.float32), mask.to(torch.float32)
    
    def __len__(self):
        return self.dir_length