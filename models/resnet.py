import torch.nn as nn
import torchvision

class ResNet(nn.Module):
  def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=2, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.n_classes = n_classes
    models_dict = {
      'resnet18': torchvision.models.resnet18,
      'resnet34': torchvision.models.resnet34,
      'resnet50': torchvision.models.resnet50,
      'resnet101': torchvision.models.resnet101,
      'resnet152': torchvision.models.resnet152
    }
    self.base_model = models_dict[model](weights=None)
    self._feature_vector_dimension = self.base_model.fc.in_features
    self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
    self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.base_model(x)
    features = x.view(x.size(0), -1)
    x = self.fc(features)
    return x, features