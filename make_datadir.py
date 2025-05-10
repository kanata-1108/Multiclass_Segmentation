import os
import re
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

# ディレクトリの初期化用関数
def init_dir(dir_path):

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)

# *.dcmと*.pngだけを抽出してリストに格納する関数
def get_data(dicom_dir, mask_dir):

    dicom_sample_list = []
    mask_sample_list = []
    sample_names = sorted(os.listdir(dicom_dir))
    
    for sample_name in sample_names:
        dicom_samples = glob(dicom_dir + sample_name + '/*.dcm')
        mask_samples = [file for file in glob(mask_dir + sample_name + '/*.npy') if re.search(r'_segmentation_', file)]

        for dicom_sample, mask_sample in zip(dicom_samples, mask_samples):
            dicom_sample_list.append(dicom_sample)
            mask_sample_list.append(mask_sample)
    
    dicom_sample_list, mask_sample_list = sorted(dicom_sample_list), sorted(mask_sample_list)

    return dicom_sample_list, mask_sample_list

def copy_files(file_paths, out_dir):

    for file_path in file_paths:
        shutil.copy(file_path, out_dir)

if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    dcm_list, msk_list = get_data('./data_dicom/original_data/dicom/', './data_dicom/original_data/mask/')

    # 各データを格納するディレクトリの作成
    output_dir = './data_dicom'
    train_image_dir = os.path.join(output_dir, "train/dicom")
    train_mask_dir = os.path.join(output_dir, "train/mask")
    valid_image_dir = os.path.join(output_dir, "valid/dicom")
    valid_mask_dir = os.path.join(output_dir, "valid/mask")
    test_image_dir = os.path.join(output_dir, "test/dicom")
    test_mask_dir = os.path.join(output_dir, "test/mask")

    os.makedirs(train_image_dir, exist_ok = True)
    os.makedirs(train_mask_dir, exist_ok = True)
    os.makedirs(valid_image_dir, exist_ok = True)
    os.makedirs(valid_mask_dir, exist_ok = True)
    os.makedirs(test_image_dir, exist_ok = True)
    os.makedirs(test_mask_dir, exist_ok = True)

    # 初期化処理
    init_dir(train_image_dir)
    init_dir(train_mask_dir)
    init_dir(valid_image_dir)
    init_dir(valid_mask_dir)
    init_dir(test_image_dir)
    init_dir(test_mask_dir)

    train_dcm, temp_dcm, train_mask, temp_mask = train_test_split(dcm_list, msk_list, test_size = 0.2, random_state = 7)
    valid_dcm, test_dcm, valid_mask, test_mask = train_test_split(temp_dcm, temp_mask, test_size = 0.5, random_state = 7)

    copy_files(train_dcm, train_image_dir)
    copy_files(train_mask, train_mask_dir)
    copy_files(valid_dcm, valid_image_dir)
    copy_files(valid_mask, valid_mask_dir)
    copy_files(test_dcm, test_image_dir)
    copy_files(test_mask, test_mask_dir)