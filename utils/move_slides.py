# this script is used to process the slides uploaded by the program and save the processed data as well as the images
# the processed data is the original data multiplied by 255 and converted to uint8
# the images are the processed data and the original data

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

folder = '/mnt/disks/whole/temp_slides'

# Find all matching files
pattern = 'SBC_20240725_*.npy'
matching_files = glob.glob(os.path.join(folder, pattern))

for file_path in matching_files:
    # Load the file
    data = np.load(file_path)
    
    # Process: multiply by 255 and convert to uint8
    processed = (data * 255).astype(np.uint8)
    
    # Create new filename with _processed suffix
    base_path = os.path.splitext(file_path)[0]
    new_folder = './processed'
    new_path = f"{new_folder}/{os.path.basename(base_path)}.npy"
    
    # Save the processed array
    np.save(new_path, processed)
    print(f"Processed and saved: {new_path}")

    # show first image's first three channels
    frame = processed[12347, :, :, :]
    frame_original = data[12347, :, :, :]
    image = frame.transpose(1, 2, 0)
    image_original = frame_original.transpose(1, 2, 0)
    img_fluorescence = image[:, :, [2, 1, 0]]
    img_fluorescence_original = image_original[:, :, [2, 1, 0]]
    img_dpc = image[:, :, 3]
    img_dpc_original = (image_original[:, :, 3]*255).astype(np.uint8)
    img_overlay = (0.64 * img_fluorescence + 0.36 * np.dstack([img_dpc]*3)).astype('uint8')
    img_overlay_original = (0.64 * img_fluorescence_original + 0.36 * np.dstack([img_dpc_original]*3)).astype('uint8')
    print(img_overlay.shape)
    plt.imshow(img_overlay)
    plt.savefig(f"{new_folder}/{os.path.basename(base_path)}.png")
    plt.imshow(img_overlay_original)
    plt.savefig(f"{new_folder}/{os.path.basename(base_path)}_original.png")

    # save the dpc as well, multiply it to 3 channels
    img_dpc = np.dstack([img_dpc]*3)
    plt.imsave(f"{new_folder}/{os.path.basename(base_path)}_dpc.png", img_dpc)
    img_dpc_original = np.dstack([img_dpc_original]*3)
    plt.imsave(f"{new_folder}/{os.path.basename(base_path)}_dpc_original.png", img_dpc_original)
    # save the fluorescence as well
    plt.imsave(f"{new_folder}/{os.path.basename(base_path)}_fluorescence.png", img_fluorescence)
    plt.imsave(f"{new_folder}/{os.path.basename(base_path)}_fluorescence_original.png", img_fluorescence_original)

    print(img_fluorescence_original)
    break
