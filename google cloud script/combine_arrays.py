import numpy as np
import os
from glob import glob

def combine_arrays(folder_path, output_dir="combined"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all filtered_spots files and sort them
    spot_files = sorted(glob(os.path.join(folder_path, '*_cropped.npy')))
    
    # Load first arrays to get shapes
    first_spots = np.load(spot_files[0])
    
    print(f"Individual array shapes:")
    print(f"Filtered spots: {first_spots.shape}")
    
    # Initialize lists to store all arrays
    all_spots = []
    skipped_fovs = []
    
    # Load and combine all arrays
    for spot_file in spot_files:
        spots = np.load(spot_file)
        
        # Skip empty FOVs or FOVs with wrong dimensions
        if spots.size == 0 or spots.ndim != 4 or spots.shape[1:] != (4, 31, 31):
            fov_name = os.path.basename(spot_file).split('_')[0]
            skipped_fovs.append(fov_name)
            continue
            
        all_spots.append(spots)
    
    if skipped_fovs:
        print(f"\nSkipped {len(skipped_fovs)} empty FOVs: {', '.join(skipped_fovs)}")
    
    # Concatenate along first axis
    combined_spots = np.concatenate(all_spots, axis=0)
    
    print(f"\nCombined array shapes:")
    print(f"Combined spots: {combined_spots.shape}")
    
    # Convert to uint8
    combined_spots = (combined_spots * 255).astype(np.uint8)
    
    # Save combined arrays in the output directory
    folder_name = os.path.basename(folder_path)
    np.save(os.path.join(output_dir, f"{folder_name}.npy"), combined_spots)
    print(f"\nSaved combined array in {output_dir}/ as {folder_name}.npy")

if __name__ == "__main__":
    # Get all slide folders in downloaded_data
    data_dir = "/mnt/disks/whole/new_sbcs/"
    slide_folders = sorted(glob(os.path.join(data_dir, "SBC_*")))
    
    print(f"Found {len(slide_folders)} slide folders to process")
    
    # Process each slide folder
    for folder in slide_folders:
        print(f"\nProcessing {os.path.basename(folder)}...")
        combine_arrays(folder, output_dir="/mnt/disks/whole/rep_sbcs") 