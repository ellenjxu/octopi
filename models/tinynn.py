# based on https://en.wikipedia.org/wiki/LeNet

import torch.nn as nn
import torch.nn.functional as F

class TinyNN(nn.Module):
  def __init__(self, n_channels=4, n_classes=2): 
    super().__init__()
    self.conv1 = nn.Conv2d(n_channels, 6, kernel_size=5, stride=1, padding=2)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # check
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, n_classes)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x