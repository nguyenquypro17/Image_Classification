import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)    

class BasicBlock(nn.Module):
    def __init__(self, inplane, plane, stride = 1): # inplane: số kênh đầu vào block, plane: số kênh của conv trong block
        super(BasicBlock, self).__init__() 
        self.conv1 = nn.Conv2d(inplane, plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)
        self.shortcut = nn.Sequential()
        if stride == 2 or inplane != plane:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplane, plane, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(plane)
            ) 

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class Resnet(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.inplane = 16

        self.layer1 = self._make_layer(block, 16, num_block[0], stride = 1) # 16 số channel của conv
        self.layer2 = self._make_layer(block, 32, num_block[1], stride = 2) # stride = 2 giảm kích thước ảnh đầu mỗi stage
        self.layer3 = self._make_layer(block, 64, num_block[2], stride = 2) # stride = 2 giảm kích thước ảnh đầu mỗi stage

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, num_classes)
        self.apply(weight_init)

    def _make_layer(self, block, plane, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, plane, stride))
            self.inplane = plane
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
def Resnet20():
    return Resnet(BasicBlock, [3, 3, 3]) 

def Resnet56():
    return Resnet(BasicBlock, [9, 9, 9])

def Resnet110():
    return Resnet(BasicBlock, [18, 18, 18])

