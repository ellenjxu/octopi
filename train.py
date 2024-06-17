"""
train-val pipeline
"""

import hydra
from hydra.utils import instantiate
import wandb
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os

from utils.dataset import TrainDataset, get_transforms, collate_transforms
device ='cuda' if torch.cuda.is_available() else 'cpu'

# check cuda availability
print("Device: ", device)

def get_outputs(model, dl, criterion=None):
  scores, labels, losses = [], [], []
  model = model.eval()

  with torch.no_grad():
    for images, label in tqdm(dl):
      images, label = images.to(device), label.to(device)
      pred = model(images)

      if criterion is not None:
        loss = criterion(pred, label)
        losses.append(loss.cpu())

      probs = torch.softmax(pred, dim=1) # normalized
      scores.append(probs.cpu())
      labels.append(label.cpu())
  
  scores = torch.cat(scores, dim=0)
  labels = torch.cat(labels, dim=0)
  
  return scores, labels, losses

@hydra.main(config_path="config/", config_name="config_gcloud", version_base="1.1")
def main(cfg):
  if cfg.wandb.enabled:
    config = OmegaConf.to_container(
      cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, save_code=True, job_type=cfg.wandb.job_type, config=config)

  ds = TrainDataset(data_dir=cfg.train.data_dir, transform=None)

  val_len = int(cfg.train.val_split * len(ds))
  train_len = len(ds) - val_len
  train_ds, val_ds = random_split(ds, [train_len, val_len])
  print(f"train: {train_len}, val: {val_len}")
  
  # train_transform = get_transforms(augment=False) # TODO: test augments
  # val_transform = get_transforms(augment=False)
   
  train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                            # collate_fn=lambda x: collate_transforms(x, train_transform),
                            num_workers=cfg.train.num_workers,
                            pin_memory=True)
  val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, 
                          # collate_fn=lambda x: collate_transforms(x, val_transform),
                          num_workers=cfg.train.num_workers,
                          pin_memory=True)

  model = instantiate(cfg.model).to(device)
  criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.train.class_weights, device=device))
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

  best_val_loss = float("inf")
  best_model = None

  for epoch in range(cfg.train.epochs):
    print(f"epoch: {epoch}")
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      
      if cfg.wandb.enabled:
        wandb.log({"train/loss": loss})
    
    avg_train_loss = total_loss / len(train_loader)

    preds, labels, losses = get_outputs(model, val_loader, criterion)
    avg_val_loss = torch.stack(losses).mean()
    avg_val_acc = accuracy_score(labels.numpy(), preds.argmax(dim=1).numpy())

    print(f"train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}, val acc: {avg_val_acc:.4f}")
    if cfg.wandb.enabled:
      wandb.log({"train/loss": avg_train_loss, "val/loss": avg_val_loss, "val/acc": avg_val_acc})

    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      best_model = model.state_dict()
      print("best model updated at epoch: ", epoch)

  if cfg.train.save_model:
    out_dir = os.path.join(cfg.train.out_dir, cfg.wandb.name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    torch.save(best_model, os.path.join(out_dir, "best.pt"))

  if cfg.wandb.enabled:
    artifact = wandb.Artifact(f"{cfg.wandb.name}", type="model")
    artifact.add_file(os.path.join(out_dir, "model.pt"))
    artifact.add_file(os.path.join(out_dir, "best.pt"))
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    main()
