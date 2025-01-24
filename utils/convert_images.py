import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
import json
from tqdm import tqdm

def preprocess_images(npy_file_path, base_output_dir):
    # Extract dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(npy_file_path))[0]
    
    # Load the numpy array
    npy_data = np.load(npy_file_path)
    
    # Process all images in the dataset
    processed_data = {}
    for i, frame in enumerate(npy_data):
        frame = frame.transpose(1, 2, 0)
        img_fluorescence = frame[:, :, [2, 1, 0]]
        img_dpc = frame[:, :, 3]
        img_overlay = (0.64 * img_fluorescence + 0.36 * np.dstack([img_dpc]*3)).astype('uint8')
        
        # Convert to PIL Image
        img = Image.fromarray(img_overlay, 'RGB')
        
        # Save as BMP in memory
        buffer = BytesIO()
        img.save(buffer, format="BMP")
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Store in dictionary
        processed_data[i] = img_base64

    # Save processed data as JSON
    output_file = os.path.join(base_output_dir, f'{dataset_name}.json')
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)

    print(f"Preprocessed {len(processed_data)} images for dataset {dataset_name} to {output_file}")

# Usage
base_output_dir = '/mnt/disks/images/base64'
npy_folder_path = '/mnt/disks/whole/whole-slides/'

for npy_file in tqdm(os.listdir(npy_folder_path)):
    if npy_file.endswith('.npy'):
        npy_file_path = os.path.join(npy_folder_path, npy_file)
        preprocess_images(npy_file_path, base_output_dir)