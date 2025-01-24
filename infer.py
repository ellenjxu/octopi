import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from utils.dataset import SinglePatientDataset
import os
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_outputs(model, dl):
    """Get model outputs for a dataloader"""
    scores, labels, _, features = [], [], [], []
    model = model.eval()

    with torch.no_grad():
        for images, label in dl:
            images, label = images.to(device), label.to(device)
            pred, feature = model(images)
            probs = torch.softmax(pred, dim=1)
            scores.append(probs.cpu())
            labels.append(label.cpu())
            features.append(feature.cpu())
    
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)
    features = torch.cat(features, dim=0)
    
    return scores, labels, features

def infer_file(model, npy_filepath, out_dir, batch_size=32, num_workers=4, save_features=False):
    """
    Run inference on a single .npy file and save predictions to csv
    
    Args:
        model: PyTorch model
        npy_filepath: Path to input .npy file
        out_dir: Directory to save output csv
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        save_features: Whether to save feature vectors
        
    Returns:
        Path to saved csv file
    """
    # Setup model
    model = model.to(device)
    model.eval()
    
    # Create output directory if needed
    os.makedirs(out_dir, exist_ok=True)
    if save_features:
        feature_dir = os.path.join(out_dir, "features")
        os.makedirs(feature_dir, exist_ok=True)
    
    # Get dataset ID from filename
    dataset_id = npy_filepath.name.split("_cleaned")[0].split(".npy")[0]
    
    # Setup data loading
    test_ds = SinglePatientDataset(npy_filepath)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Run inference
    probs, labels, features = get_outputs(model, test_loader)
    labels = labels.numpy()

    # Create output dataframe
    output_df = pd.DataFrame({
        'index': range(len(probs)),
        'non-parasite output': probs[:,0],
        'parasite output': probs[:,1],
        'label': labels
    })

    # Save features if requested
    if save_features:
        features = features.numpy()
        features = features.reshape(features.shape[0], -1)  # flatten
        feature_file = os.path.join(feature_dir, f"{dataset_id}.npy")
        np.save(feature_file, features)

    # Save predictions
    csv_path = os.path.join(out_dir, f"{dataset_id}.csv")
    output_df.to_csv(csv_path, index=False)
    
    return csv_path

if __name__ == "__main__":
    import hydra
    import os
    from pathlib import Path
    
    @hydra.main(config_path="config/", config_name="config_gcloud", version_base="1.1")
    def main(cfg):
        # Load model
        model = hydra.utils.instantiate(cfg.model).to(device)
        model_path = os.path.join(cfg.train.out_dir, cfg.wandb.name, cfg.test.cp_name)
        model.load_state_dict(torch.load(model_path))
        
        # Setup output directory
        out_dir = os.path.join(cfg.test.out_dir, cfg.wandb.name, "csv_whole" if cfg.test.whole else "csv")
        
        # Get input file path from hydra config or command line
        try:
            npy_path = Path(cfg.npy_file)
        except:
            print("Usage: python infer.py npy_file=/path/to/file.npy")
            sys.exit(1)
            
        if not npy_path.exists():
            print(f"Error: File {npy_path} does not exist")
            sys.exit(1)
            
        csv_path = infer_file(
            model=model,
            npy_filepath=npy_path,
            out_dir=out_dir,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            save_features=cfg.test.save_features
        )
        print(f"Saved predictions to {csv_path}")

    main()