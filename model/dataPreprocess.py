import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, in_dir, out_dir, transform=None):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.transform = transform

        self.in_images = sorted(os.listdir(in_dir))
        self.out_images = sorted(os.listdir(out_dir))


        assert len(self.in_images) == len(self.out_images), "输入和输出文件夹中的图片数量不一致"

    def __len__(self):
        return len(self.in_images)

    def __getitem__(self, idx):
        in_image_path = os.path.join(self.in_dir, self.in_images[idx])
        out_image_path = os.path.join(self.out_dir, self.out_images[idx])

        in_image = Image.open(in_image_path).convert('L')  # 转换为灰度图像
        out_image = Image.open(out_image_path).convert('L')  # 转换为灰度图像

        if self.transform:
            in_image = self.transform(in_image)
            out_image = self.transform(out_image)

        in_image = in_image.permute(1, 2, 0)  # Change from [1, 256, 256] to [256, 256, 1]
        out_image = out_image.permute(1, 2, 0)  
        return in_image, out_image

def get_data_loader(in_dir, out_dir, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256,256)),  # Resize to 256x256
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(in_dir, out_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

# 使用示例
in_dir = '/home/cxy_0/ZernikePro/targetRealScence/085Ma_Azi0_Elv60_Cn001/pic/in'
out_dir = '/home/cxy_0/ZernikePro/targetRealScence/085Ma_Azi0_Elv60_Cn001/pic/out'
batch_size = 8

data_loader = get_data_loader(in_dir, out_dir, batch_size=batch_size)

'''
# 打印验证对应的dataLoader形状

for in_images, out_images in data_loader:
    print(f"Input batch shape: {in_images.shape}")
    print(f"Output batch shape: {out_images.shape}")
    # 执行模型训练或验证
'''