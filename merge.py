import pandas as pd
import os
from tqdm import tqdm
from utils.viz import merge

model = "resnet18"
ver1 = "retrain_v1"
ver2 = "retrain_v3"

merge(model, ver1, ver2)

