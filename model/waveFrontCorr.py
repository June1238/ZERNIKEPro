import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16
from nonLocalBlock import NONLocalBlock2D
from selfAttention import Self_Attn
from transformer import ImageTransformer


class waveCorrNet(nn.Module):
    def __init__(self):
        super(waveCorrNet  , self).__init__()
        
        self.nonLocalBlock = NONLocalBlock2D(1, sub_sample=False, bn_layer=True)
        self.attnBlock = Self_Attn(in_dim=1, activation=nn.ReLU())
        self.trans = ImageTransformer(patch_size=4, in_channels=1, embed_dim=64, num_heads=4, feedforward_dim=256, num_layers=4, dropout=0.1)


        # Load the pretrained ResNet-50 model
        resnet = resnet50(pretrained=True)
        self.resnet_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  # Convert 1-channel input to 3-channel
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # Include the average pooling layer
        )
        
        # Fully connected layer to adjust the output size to 140
        self.fc = nn.Linear(resnet.fc.in_features, 140)
        

    def forward(self, x1, x2):
        # Ensure x1 and x2 have the shape [batch_size, 256, 256, 1]
        x1 = x1.permute(0, 3, 1, 2)  # Change shape from [batch_size, 256, 256, 1] to [batch_size, 1, 256, 256]
        x2 = x2.permute(0, 3, 1, 2)  # Change shape from [batch_size, 256, 256, 1] to [batch_size, 1, 256, 256]
        
        # Concatenate x1 and x2 along the height dimension
        x = torch.cat((x1, x2), dim=2)  
        print(x.shape)


        out = self.trans(x)
        # out,_ = self.attnBlock(x)
        out = self.resnet_layer(out)  # Pass through the ResNet layers
        print("Pass through ResNet layers completed")

        out = torch.flatten(out, 1)  # Flatten the tensor
        out = self.fc(out)  # Fully connected layer to get the desired output size

        return out

'''
# Specify device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Instantiate model and move to device
model = CustomModel().to(device)

# Example usage
batch_size = 8
x1 = torch.randn(batch_size, 256, 256, 1).to(device)  # Example input tensor with shape [batch_size, 256, 256, 1]
x2 = torch.randn(batch_size, 256, 256, 1).to(device)  # Example input tensor with shape [batch_size, 256, 256, 1]

# Move model and inputs to the specified device
model.to(device)
x1 = x1.to(device)
x2 = x2.to(device)

# Forward pass
output = model(x1, x2)
print(output.shape)
'''