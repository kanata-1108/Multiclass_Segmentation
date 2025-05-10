import torch
import numpy as np
from PIL import Image

# 入力: 256x256 の Tensor（各ピクセルが 0,1,2,3 のいずれかのクラス）
segmentation = torch.randint(0, 4, (256, 256))  # 例としてランダム生成
print(segmentation.shape)

# 各クラスに対応する RGB カラーを定義
# 例: 0 → 黒, 1 → 赤, 2 → 緑, 3 → 青
palette = {
    0: (0, 0, 0),       # black
    1: (255, 0, 0),     # red
    2: (0, 255, 0),     # green
    3: (0, 0, 255)      # blue
}

# RGB チャンネルごとの画像を作成
segmentation_np = segmentation.numpy()
height, width = segmentation_np.shape
color_image = np.zeros((height, width, 3), dtype=np.uint8)

for class_id, color in palette.items():
    mask = segmentation_np == class_id
    color_image[mask] = color

# PIL で表示・保存も可能
img = Image.fromarray(color_image)
# img.show()  # 表示
# img.save('/home/kotori/src/Multiclass_Segmentation/segmentation_colored.png')  # 保存したい場合