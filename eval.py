"""
eval pipeline
- calculates FNR for 5 LOD threshold for each csv
- plot FNR vs FP ratio matrix
"""

import hydra
import pandas as pd
import os
import torch

from utils.viz import *

device ='cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="config/", config_name="config", version_base="1.1")
def main(cfg):
  # TODO: Currently picks lowest possible threshold for LOD, doesn’t handle well when there are less than 5 FP / no high positive prediction scores
  plot_threshold(cfg.evaluate.csv_dir, cfg.evaluate.out_dir)

  # fpr and fnr
  fpr_list, fnr_list = calculate_fpr_fnr(cfg.evaluate.csv_dir, threshold=0.5)
  print(f"FPR: {np.mean(fpr_list):.4f}, FNR: {np.mean(fnr_list):.4f}")

  # ratio matrix
  plot_ratio_matrix(cfg.evaluate.csv_dir, cfg.evaluate.out_dir)
  
if __name__ == "__main__":
  main()
