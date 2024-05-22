import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DWConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)

        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size # 小块的大小，如(4, 4)
        self.in_channels = in_channels # 输入通道数，如3
        self.embed_dim = embed_dim # 嵌入向量的维度，如64

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # x的形状变为(batch_size, embed_dim, height//patch_size, width//patch_size)，如(1, 64, 8, 8)
        x = x.flatten(2) # x的形状变为(batch_size, embed_dim, height//patch_size * width//patch_size)，如(1, 64, 64)
        x = x.transpose(1, 2) # x的形状变为(batch_size, height//patch_size * width//patch_size, embed_dim)，如(1, 64, 64)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim # 嵌入向量的维度，如64
        self.num_heads = num_heads # 多头注意力的头数，如8
        self.feedforward_dim = feedforward_dim # 前馈神经网络的隐层维度，如256
        self.dropout = dropout # dropout的概率，如0.1
        self.a = 16

        # 使用nn.MultiheadAttention来实现多头自注意力机制
        # 这里为了配合QKV变换，需要将embd乘2
        self.attn = nn.MultiheadAttention(2*embed_dim ,num_heads ,dropout=dropout,batch_first=True)

        # self.attn = nn.MultiheadAttention(embed_dim ,num_heads ,dropout=dropout,batch_first=True)

        # self.conv1d = nn.Conv1d(4096,4096//16,kernel_size=3,padding=1)
        # self.relative_pos = nn.Parameter(torch.randn(
        #    self.num_heads, 256//self.patch_size, 256//self.patch_size//8//8))
        # self.attn = LMHSAttention(embed_dim, num_heads,sr_ratio=8)

        # 使用两个全连接层和一个激活函数来实现前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim ,feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim ,embed_dim)
        )
        # self.down_sample=nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 使用nn.LayerNorm来实现层归一化
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        # 定义一个线性层，将Q变换为b*m*(2n)
        self.linear_Q = nn.Linear(embed_dim, 2*embed_dim)
        self.linear_attn = nn.Linear(2*embed_dim, embed_dim)
        # 定义一个1D卷积层，将K变换为b*(m/a)*(2n)，其中a是卷积的步长
        self.conv_K = nn.Conv1d(embed_dim, 2*embed_dim, kernel_size=1, stride=self.a)
        # 定义一个1D卷积层，将V变换为b*(m/a)*(2n)，其中a是卷积的步长
        self.conv_V = nn.Conv1d(embed_dim, 2*embed_dim, kernel_size=1, stride=self.a)
        


    def forward(self ,x):

        # 多头自注意力机制
        # x.shape (2,4096,64)
        # print(x.shape)
        # 变换Q
        Q = self.linear_Q(x)
        # 变换K
        K = self.conv_K(x.transpose(1, 2)) # b*n*m -> b*(2n)*(m/a)
        K = K.transpose(1, 2) # b*(m/a)*(2n)
        # 变换V
        V = self.conv_V(x.transpose(1, 2)) # b*n*m -> b*n*(m/a)
        V = V.transpose(1, 2) # b*(m/a)*n
        
        # print(x2.shape,x3.shape)
        attn_out ,_ = self.attn(Q ,K ,V) # attn_out的形状为(batch_size ,seq_len ,embed_dim)，如(1 ,64 ,64)
        # print(attn_out.shape)
        # LHMSA 关键：对X进行处理,K,V降维处理为K' V'
        # 此时 attn_out需要变回来
        attn_out = self.linear_attn(attn_out)
        # print("123:",attn_out.shape)
        # return

        attn_out = F.dropout(attn_out ,p=self.dropout) # 对attn_out进行dropout
        x = x + attn_out # 残差连接
        x = self.ln1(x) # 层归一化

        # 前馈神经网络
        ffn_out = self.ffn(x) # ffn_out的形状为(batch_size ,seq_len ,embed_dim)，如(1 ,64 ,64)
        ffn_out = F.dropout(ffn_out) # 对ffn_out进行dropout
        x = x + ffn_out # 残差连接
        x = self.ln2(x) # 层归一化

        return x

# 定义一个图像Transformer模型，包含一个Patch Embedding层和多个Transformer Encoder层，并将输出重构成图像形状
class ImageTransformer(nn.Module):
    def __init__(self ,patch_size ,in_channels ,embed_dim ,num_heads ,feedforward_dim ,num_layers ,dropout):
        super().__init__()
        self.patch_size = patch_size # 小块的大小，如(4, 4)
        self.in_channels = in_channels # 输入通道数，如3
        self.embed_dim = embed_dim # 嵌入向量的维度，如64
        self.num_heads = num_heads # 多头注意力的头数，如8
        self.feedforward_dim = feedforward_dim # 前馈神经网络的隐层维度，如256
        self.num_layers = num_layers # Transformer Encoder层的个数，如4
        self.dropout = dropout # dropout的概率，如0.1


        # 使用Patch Embedding层来将图像分割成小块，并将每个小块映射成一个向量
        self.patch_embed = PatchEmbedding(patch_size ,in_channels ,embed_dim)

        
        # Local Perception Unit
        self.LPU = DWConv(in_channels, in_channels)

        # 使用transFormer头来提取特征
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim ,num_heads ,feedforward_dim ,dropout) for _ in range(num_layers)
        ])

        # 使用一个卷积层来实现输出重构，将向量映射回图像
        self.reconstruct = nn.ConvTranspose2d(embed_dim ,in_channels ,kernel_size=patch_size ,stride=patch_size)
 
    def forward(self, x):
        # x的形状为(batch_size ,in_channels ,height ,width)，如(1 ,3 ,32 ,32)
        c = x.shape[1]
        h = x.shape[2] 
        w = x.shape[3]

        # Patch Embedding
        # print(x.shape)
        
        x = self.LPU(x) + x
        x = self.patch_embed(x) # x的形状为(batch_size ,height//patch_size * width//patch_size ,embed_dim)，如(1 ,64 ,64)
        
        # print(x.shape)
        # return
        # Transformer Encoder层
        
        for layer in self.encoder_layers:
            x = layer(x) # x的形状不变，如(1 ,64 ,64)

        # 输出重构
        x = x.transpose(1 ,2) # x的形状变为(batch_size ,embed_dim ,height//patch_size * width//patch_size)，如(1 ,64 ,64)
        x = x.reshape(x.shape[0] ,x.shape[1] ,-1 ,self.patch_size) # x的形状变为(batch_size ,embed_dim ,height//patch_size ,width//patch_size)，如(1 ,64 ,8 ,8)
        x = self.reconstruct(x) # x的形状变为(batch_size ,in_channels ,height ,width)，如(1 ,3 ,32 ,32)

        return x.reshape(x.shape[0] ,x.shape[1],h,w)