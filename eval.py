"""
eval pipeline
- calculates FNR for 5 LOD threshold for each csv
- plot FNR vs FP ratio matrix

outputs: thresholds/ in model folder
"""

import hydra
import pandas as pd
import os
import torch

from utils.viz import *

device ='cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="config/", config_name="config_gcloud", version_base="1.1")
def main(cfg):
  csv_dir = cfg.evaluate.csv_dir
  out_dir = cfg.evaluate.out_dir

  FP_target = 5

  # create out_dir
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  thre_start = plot_threshold(csv_dir, out_dir,FP_target=FP_target)
  plot_fp_fnr(csv_dir, out_dir, thr_start=thre_start)
  plot_ratio_matrix(csv_dir, out_dir, FP_target)
  TPR, FPR, TNR, FNR = plot_confusion_matrix(csv_dir, out_dir, threshold = thre_start)
  plot_roc_curve(csv_dir, out_dir,fpr_end = 0.0001, fpr_cutoff = FPR, tpr_cutoff = TPR)

if __name__ == "__main__":
  main()
