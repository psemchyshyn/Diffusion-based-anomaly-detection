import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self):
        super(DiscriminativeSubNetwork, self).__init__()
        # self.conv1 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(3)
        # self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(8)
        # self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(16)
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.bn3(self.conv3(x)))
        x = self.resnet(x)
        # x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sm(x)
        return x

