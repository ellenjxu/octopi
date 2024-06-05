import torch
from torch.utils.data import Dataset, default_collate
from torchvision import transforms
import os
from pathlib import Path
import numpy as np

class TrainDataset(Dataset):
  """
  loads octopi malaria Giemsa dataset
  """
  def __init__(self, data_dir, transform=None):
    self.files = list(Path(data_dir).rglob('*.npy'))
    #self.images = np.concatenate([np.load(f) for f in self.files])
    #self.labels = np.concatenate([np.ones(len(self.files)) if "pos" in f.parent.name else np.zeros(len(self.files)) for f in self.files])
    
    self.images = []
    self.labels = []
    for f in self.files:
        data = np.load(f)
        self.images.append(data)
        self.labels.append(np.ones(len(data)) if "pos" in f.parent.name else np.zeros(len(data)))
    self.images = np.concatenate(self.images)
    self.labels = np.concatenate(self.labels)
    self.transform = transform
    
  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    image = torch.tensor(image, dtype=torch.float32) / 255.0
    label = self.labels[idx]
    if self.transform:
      image = self.transform(image)
    return image, torch.tensor(label, dtype=torch.long)

class SinglePatientDataset(Dataset):
  """
  gets a single patient dataset file .npy
  """
  def __init__(self, filepath):
    self.images = np.load(filepath)
    self.labels = np.ones(len(self.images)) if "pos" in filepath.parent.name else np.zeros(len(self.images))

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    return torch.tensor(self.images[idx], dtype=torch.float32) / 255.0, torch.tensor(self.labels[idx], dtype=torch.long)

def get_transforms(augment=True):
  transform = [
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for imagenet
  ]
  if augment:
    transform.extend([
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=1.5),
      # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
      # transforms.RandomResizedCrop(size=(31, 31), scale=(0.8, 1.0), ratio=(0.75, 1.33))
    ])
  return transforms.Compose(transform)

def collate_transforms(batch, transform=None):
  images, labels = zip(*batch)
  if transform:
    images = [transform(i) for i in images]
  return default_collate(list(zip(images, labels))) # back into type Tensor
