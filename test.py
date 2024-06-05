"""
runs model.pt and saves patient-level preds on external test set
"""

import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils.dataset import SinglePatientDataset
from train import get_outputs
from pathlib import Path
import os

device ='cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="config/", config_name="config", version_base="1.1")
def main(cfg):
  model = hydra.utils.instantiate(cfg.model).to(device)
  model_path = os.path.join(cfg.train.out_dir, cfg.wandb.name, "model.pt")
  model.load_state_dict(torch.load(model_path))
  model.eval()
  
  out_dir = os.path.join(cfg.test.out_dir, cfg.wandb.name, "csv")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  if cfg.test.use_full_dataset:
    test_files = list(Path(cfg.test.data_dir).rglob('*.npy'))
  else:
    test_files = list(Path(cfg.test.data_dir).rglob('*.npy'))[:20]

  for filepath in test_files:
    dataset_id = filepath.name.split("_cleaned")[0].split(".npy")[0]
    test_ds = SinglePatientDataset(filepath)
    test_loader = DataLoader(test_ds, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_workers)

    probs, labels, _ = get_outputs(model, test_loader)
    labels = labels.numpy()

    output_df = pd.DataFrame({
      'index': range(len(probs)),
      'non-parasite output': probs[:,0],
      'parasite output': probs[:,1],
      'label': labels})
    output_df.to_csv(os.path.join(out_dir, f"{dataset_id}.csv"), index=False)

if __name__ == "__main__":
  main()
