import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from waveFrontCorr import waveCorrNet  
from zerNikeDataLoader import zernikeYData_loader as zernikeYDataLoader
from dataPreprocess import data_loader as zernikeXDataLoader

# 定义 RMS 函数
def rms(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

# 定义训练函数
def train(model, dataloaderx, dataloadery, criterion, optimizer, num_epochs, device, out_file):
    model.train()
    max_rms = 0.0  # 用于记录最大 RMS 值
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (x_batch, y_batch) in zip(dataloaderx, dataloadery):
            x1, x2 = x_batch[0], x_batch[1]
            x1, x2, y_label = x1.to(device), x2.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x1, x2)
            loss = criterion(y_pred, y_label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloaderx)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 每20轮计算并输出一次 RMS
        if (epoch + 1) % 20 == 0:
            avg_rms = test(model, dataloaderx, dataloadery, criterion, device, out_file, write_to_file=False)
            if avg_rms > max_rms:
                max_rms = avg_rms
            print(f'Epoch [{epoch + 1}/{num_epochs}], RMS: {avg_rms:.4f}, Max RMS: {max_rms:.4f}')
            
            # 将RMS和Loss写入文件
            with open(out_file, 'a') as f:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, RMS: {avg_rms:.4f}\n')
    
    # 将最大 RMS 写入文件
    with open(out_file, 'a') as f:
        f.write(f'Max RMS: {max_rms:.4f}\n')

# 定义测试函数
def test(model, dataloaderx, dataloadery, criterion, device, out_file, write_to_file=True):
    model.eval()
    total_rms = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for (x_batch, y_batch) in zip(dataloaderx, dataloadery):
            x1, x2 = x_batch[0], x_batch[1]
            x1, x2, y_label = x1.to(device), x2.to(device), y_batch.to(device)
            
            y_pred = model(x1, x2)
            loss = criterion(y_pred, y_label)
            batch_rms = rms(y_pred, y_label)
            
            total_rms += batch_rms.item()
            total_loss += loss.item()
    
    avg_rms = total_rms / len(dataloaderx)
    avg_loss = total_loss / len(dataloaderx)
    
    if write_to_file:
        with open(out_file, 'a') as f:
            f.write(f'RMS: {avg_rms:.4f}, Loss: {avg_loss:.4f}\n')

    print(f'RMS: {avg_rms:.4f}, Loss: {avg_loss:.4f}')
    
    return avg_rms

# 主函数
def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = waveCorrNet().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100  # 假设你想训练100个epoch
    out_fileTrain = '/home/cxy_0/ZernikePro/targetRealScence/model/result/outputConcatTrain.txt'
    out_fileTest = '/home/cxy_0/ZernikePro/targetRealScence/model/result/outputConcatTest.txt'
    
    # 清空并创建输出文件
    for out_file in [out_fileTrain, out_fileTest]:
        if os.path.exists(out_file):
            os.remove(out_file)
        else:
            with open(out_file, 'w') as f:
                pass
    
    train(model, zernikeXDataLoader, zernikeYDataLoader, criterion, optimizer, num_epochs, device, out_fileTrain)
    final_rms = test(model, zernikeXDataLoader, zernikeYDataLoader, criterion, device, out_fileTest)
    print(f'Final RMS: {final_rms:.4f}')

if __name__ == '__main__':
    main()
