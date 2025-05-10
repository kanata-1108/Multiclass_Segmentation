import numpy as np
from PIL import Image
from glob import glob
import os

if __name__ == "__main__":

    files = sorted(glob('./data_dicom/test/mask/*.npy'))
    
    for file in files:
        npy = np.load(file)
        img = Image.fromarray(npy, mode = "L")
        img.save(f"{file[:-4]}" + ".png")