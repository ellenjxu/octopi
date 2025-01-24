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
import numpy as np

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

  cell_count_df = pd.read_csv('utils/cell_count.csv')
  neg_files = cell_count_df[cell_count_df['ML'] == 'Test_neg']['dataset ID'].tolist()
  negs_txt = [file + '.npy' for file in neg_files]
  negs = [x for x in whole if any([y in x.name for y in negs_txt])]

  pos_files = cell_count_df[cell_count_df['ML'] == 'Test_pos_spot']['dataset ID'].tolist()
  print(postivs)
  print(pos_files)
  pos_txt = [file + '.npy' for file in pos_files]
  pos = [x for x in postivs if any(pos_file in x.name.split('_cleaned')[0] for pos_file in pos_files)]

  print('\n'.join([x.name for x in pos]))

  if not cfg.test.whole:
    files = pos + negs
    print(f"Positives: {len(pos)}, Negatives: {len(negs)}")
  else:
    out_dir = os.path.join(cfg.test.out_dir, cfg.wandb.name, "csv_whole" if cfg.test.whole else "csv")
    # if directory does not exist, create it
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    files = whole
    print(f"Whole slides: {len(whole)}")
  
  for filepath in files: 
    dataset_id = filepath.name.split("_cleaned")[0].split(".npy")[0]
    if os.path.exists(os.path.join(out_dir, f"{dataset_id}.csv")):
      if os.path.exists(os.path.join(out_dir_features, f"{dataset_id}.npy")) or not cfg.test.save_features:
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
        feature_file = os.path.join(out_dir_features, f"{dataset_id}.npy")
        np.save(feature_file, features)

        # check if the csv already exists
    if os.path.exists(os.path.join(out_dir, f"{dataset_id}.csv")):
      print(f"Skipping {dataset_id} predictions")
      
    else:
      output_df.to_csv(os.path.join(out_dir, f"{dataset_id}.csv"), index=False)

if __name__ == "__main__":
  main()
