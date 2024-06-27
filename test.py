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

@hydra.main(config_path="config/", config_name="config_gcloud", version_base="1.1")
def main(cfg):
  model = hydra.utils.instantiate(cfg.model).to(device)
  model_path = os.path.join(cfg.train.out_dir, cfg.wandb.name, cfg.test.cp_name)
  print(os.getcwd())
  model.load_state_dict(torch.load(model_path))
  model.eval()
  
  out_dir = os.path.join(cfg.test.out_dir, cfg.wandb.name, "csv")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  out_dir_features = os.path.join(cfg.test.out_dir, cfg.wandb.name, "features")
  if not os.path.exists(out_dir_features):
    os.makedirs(out_dir_features)

  postivs = list(Path(cfg.test.data_dir_pos).rglob('*.npy'))
  whole = list(Path(cfg.test.data_dir_whole).rglob('*.npy'))
  
  with open(cfg.test.data_txt_neg, 'r') as f:
    negs_txt = f.readlines()
  negs_txt = [x.strip().replace('.csv', '.npy') for x in negs_txt]
  negs = [x for x in whole if any([y in x.name for y in negs_txt])]

  

  if not cfg.test.whole:
    files = postivs + negs
    print(f"Positives: {len(postivs)}, Negatives: {len(negs)}")
  else:
    out_dir = os.path.join(cfg.test.out_dir, cfg.wandb.name, "csv_whole")
    files = whole
    print(f"Whole slides: {len(whole)}")
  
  for filepath in files: 
    dataset_id = filepath.name.split("_cleaned")[0].split(".npy")[0]

    if os.path.exists(os.path.join(out_dir_features, f"{dataset_id}.npy")):
      if os.path.exists(os.path.join(out_dir, f"{dataset_id}.csv")):
        print(f"Skipping {dataset_id}")
        continue
    
    test_ds = SinglePatientDataset(filepath)
    test_loader = DataLoader(test_ds, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_workers)

    probs, labels, _ ,features = get_outputs(model, test_loader)
    labels = labels.numpy()

    output_df = pd.DataFrame({
      'index': range(len(probs)),
      'non-parasite output': probs[:,0],
      'parasite output': probs[:,1],
      'label': labels})

    if cfg.test.save_features:

      # if already exists, skip
      if os.path.exists(os.path.join(out_dir_features, f"{dataset_id}.npy")):
        print(f"Skipping {dataset_id} features")
      else:
        # save to numpy array
        print(f"Saving features for {dataset_id}")
        features = features.numpy()
        features = features.reshape(features.shape[0], -1)  # flatten
        features.tofile(os.path.join(out_dir_features, f"{dataset_id}.npy"))

        # check if the csv already exists
    if os.path.exists(os.path.join(out_dir, f"{dataset_id}.csv")):
      print(f"Skipping {dataset_id} predictions")
      
    else:
      output_df.to_csv(os.path.join(out_dir, f"{dataset_id}.csv"), index=False)

if __name__ == "__main__":
  main()
