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

  # create out_dir
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # best_threshold = plot_roc_curve(csv_dir, out_dir) # TODO: from val
  # print(f"Best threshold: {best_threshold:.4f}")
  # fpr_list, fnr_list = calculate_fpr_fnr(csv_dir, threshold=best_threshold)
  # print(f"FPR: {np.mean(fpr_list):.4f}, FNR: {np.mean(fnr_list):.4f}")

  plot_threshold(csv_dir, out_dir)
  plot_ratio_matrix(csv_dir, out_dir) # TODO: threshold
  plot_confusion_matrix(csv_dir, out_dir, threshold=0.001)

if __name__ == "__main__":
  main()
