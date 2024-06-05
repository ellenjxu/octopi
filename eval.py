"""
eval pipeline
1. calculates FNR for 5 LOD threshold for each csv
2. plot FNR vs FP
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
  # thresholds_df = calculate_threshold(cfg.evaluate.csv_dir, cfg.evaluate.neg_txt, cfg.evaluate.cell_count, cfg.evaluate.out_dir)
  # plot_threshold(thresholds_df, cfg.evaluate.out_dir)

  # fpr and fnr
  fpr_list, fnr_list = calculate_fpr_fnr(cfg.evaluate.csv_dir, cfg.evaluate.neg_txt, threshold=0.5)
  print(f"FPR: {np.mean(fpr_list):.4f}, FNR: {np.mean(fnr_list):.4f}")
  
if __name__ == "__main__":
  main()
