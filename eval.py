"""
eval pipeline
- calculates FNR for 5 LOD threshold for each csv
- plot FNR vs FP ratio matrix

outputs to out/model/thresholds/
"""

import hydra
import wandb
from omegaconf import OmegaConf
import pandas as pd
import os
import torch

from utils.viz import *

device ='cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="config/", config_name="config", version_base="1.1")
def main(cfg):
  if cfg.evaluate.log:
    config = OmegaConf.to_container(
      cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, save_code=True, job_type=cfg.wandb.job_type, config=config)

  csv_dir = cfg.evaluate.csv_dir
  out_dir = cfg.evaluate.out_dir
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  print(f"input folder: {csv_dir}, output folder: {out_dir}")

  #-----------------------------------------------------------
  # best_threshold = plot_roc_curve(csv_dir, out_dir) # TODO: from val
  # print(f"Best threshold: {best_threshold:.4f}")
  fpr_list, fnr_list = calculate_fpr_fnr(csv_dir, threshold=0.001)
  print(f"FPR: {np.median(fpr_list):.4f}, FNR: {np.median(fnr_list):.4f}")

  # plot_threshold(csv_dir, out_dir)
  # plot_ratio_matrix(csv_dir, out_dir)
  # plot_confusion_matrix(csv_dir, out_dir, threshold=0.001)

  plot_incorrect(csv_dir, cfg.test.data_dir, out_dir, threshold=0.001)

  #-----------------------------------------------------------

  if cfg.evaluate.log: # save evaluations
    artifact = wandb.Artifact(name="evaluations", type="output")
    artifact.add_dir(out_dir)
    artifact.add_dir(csv_dir)
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
  main()
