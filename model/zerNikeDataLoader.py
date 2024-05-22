import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def parse_zernike(file_name):
    with open(file_name, "r") as file:
        zernikes = file.read().split()
        degree = int(zernikes[1])
        zernikes = np.array(zernikes[-degree:]).astype(float)
    return zernikes

class ZernikeDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        zernikes = parse_zernike(file_name)
        return torch.FloatTensor(zernikes)  # Shape: [140]

def get_data_loader(folder_path, batch_size=8, shuffle=True):
    dataset = ZernikeDataset(folder_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return data_loader

# 假设文件夹路径是 'data/zernikes'
folder_path = '/home/cxy_0/ZernikePro/targetRealScence/085Ma_Azi0_Elv60_Cn001/zernike_jihe'


# 创建数据加载器
zernikeYData_loader = get_data_loader(folder_path, batch_size= 8, shuffle=True)

'''
# 测试数据加载器 打印dataLoader的形状
for batch in data_loader:
    print(batch.shape)  # Expected shape: [batch_size, 140]
'''