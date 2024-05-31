"""
runs model.pt and saves preds on external test set to predictions.csv
"""

import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils.dataset import ImageDataset
from train import get_outputs

device ='cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="config/", config_name="config", version_base="1.1")
def main(cfg):
  model = hydra.utils.instantiate(cfg.model).to(device)
  model.load_state_dict(torch.load(cfg.train.out_dir + "/" + cfg.wandb.name + ".pt"))
  model.eval()

  test_ds = ImageDataset(root=cfg.dataset.root, split="test", transform=None)
  print(f"test set size: {len(test_ds)}")
  test_loader = DataLoader(test_ds, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)

  probs, labels, _ = get_outputs(model, test_loader)
  preds = probs.argmax(dim=1).numpy()
  labels = labels.numpy()
  output_df = pd.DataFrame({'pos_prob': probs[:,1], 'prediction': preds, 'label': labels}) # TODO: match with patient slide id
  output_df.to_csv(f"{cfg.train.out_dir}/{cfg.wandb.name}_preds.csv", index=False)

if __name__ == "__main__":
  main()
