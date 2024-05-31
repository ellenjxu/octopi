import hydra
from hydra.utils import instantiate
import wandb
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
# from skorch import NeuralNetClassifier
# from skorch.helper import predefined_split
# from skorch.callbacks import Checkpoint
from tqdm import tqdm

from dataset import ImageDataset, get_transforms, collate_transforms
device ='cuda' if torch.cuda.is_available() else 'cpu'

def get_outputs(model, dl, criterion=None):
  preds = []
  labels = []
  losses = []

  model = model.eval()

  with torch.no_grad():
      model = model.eval()
      for images, label in dl:
        images, label = images.to(device), label.to(device)
        pred = model(images)

        if criterion is not None:
          loss = criterion(pred, label)
          losses.append(loss.cpu())

        preds.append(pred.cpu())
        labels.append(label.cpu())

  model = model.train()

  return preds, labels, losses

@hydra.main(config_path="config/", config_name="config", version_base="1.1")
def main(cfg):
  if cfg.wandb.enabled:
    config = OmegaConf.to_container(
      cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(project=cfg.wandb.project, save_code=True, job_type=cfg.wandb.job_type, config=config)

  ds = ImageDataset(root=cfg.dataset.root, split="train", transform=None)
  test_ds = ImageDataset(root=cfg.dataset.root, split="test", transform=None)

  val_len = int(cfg.dataset.val_split * len(ds))
  train_len = len(ds) - val_len
  train_dataset, val_dataset = random_split(ds, [train_len, val_len])
  
  # train_transform = get_transforms(augment=False) # TODO: test augments
  # val_transform = get_transforms(augment=False)
   
  # use collate to apply on the fly; want for train only
  train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True,
                            # collate_fn=lambda x: collate_transforms(x, train_transform),
                            num_workers=cfg.dataset.num_workers,
                            pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, 
                          # collate_fn=lambda x: collate_transforms(x, val_transform),
                          num_workers=cfg.dataset.num_workers,
                          pin_memory=True)

  model = instantiate(cfg.model).to(device)
  criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.dataset.class_weights, device=device))
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

  # net = NeuralNetClassifier(
  #   module=model,
  #   criterion=criterion,
  #   optimizer=optimizer,
  #   lr=cfg.train.lr,
  #   batch_size=cfg.dataset.batch_size,
  #   max_epochs=cfg.train.epochs,
  #   device=device,
  #   train_split=predefined_split(val_dataset),  # using the validation dataset to validate
  #   verbose=2,
  #   callbacks=[Checkpoint(dirname=cfg.wandb.project, monitor='valid_loss', mode='min')]
  # ) 

  # net.fit(train_dataset)

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

    # Validation loop
    _, _, losses = get_outputs(model, val_loader, criterion)
    avg_val_loss = torch.stack(losses).mean()

    print(f"\nEpoch {epoch+1}/{cfg.train.epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    if cfg.wandb.enabled:
      wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})

  # save as .pt
  if cfg.train.save_model:
      torch.save(model.state_dict(), f"models/{cfg.wandb.name}.pt")

  if cfg.wandb.enabled:
    artifact = wandb.Artifact(f"{cfg.wandb.name}", type="model")
    artifact.add_file(f"models/{cfg.wandb.name}.pt")
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    main()
