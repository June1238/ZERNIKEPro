import torch
import torch.nn as nn

class Self_Attn(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        
        out_channels = max(1, in_dim // 8)
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channel, width, height = x.size()
        projQuery = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        projKey = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(projQuery, projKey)
        attention = self.softmax(energy)
        
        projValue = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(projValue, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, width, height)
        
        out = self.gamma * out + x
        return out, attention

# 测试用例
def test_self_attn():
    batch_size = 2
    channels = 1  # in_dim = 1
    width = height = 256
    
    # 创建一个随机输入张量
    input_tensor = torch.randn(batch_size, channels, width, height)
    
    # 创建 Self_Attn 模块实例
    self_attn = Self_Attn(in_dim=channels, activation=nn.ReLU())
    
    # 运行前向传播
    output, attention = self_attn(input_tensor)
    
    # 打印输出和注意力图的形状
    print("Output shape: ", output.shape)
    print("Attention shape: ", attention.shape)
    
    # 验证输出和注意力图的形状是否正确
    assert output.shape == (batch_size, channels, width, height)
    assert attention.shape == (batch_size, width * height, width * height)
    
    print("Self_Attn module test passed.")

# 运行测试
# test_self_attn()
