import pandas as pd
import os
from tqdm import tqdm
from utils.viz import merge

dir1 = "out"
dir2 = "out_pat104_sbc6c1e"
model1 = "resnet18"
model2 = "resnet18"
ver1 = "h7h28_v3"
ver2 = "h7_v4"

merge(dir1,dir2,model1,model2,ver1, ver2)