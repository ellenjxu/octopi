import os
import numpy as np
from umap import UMAP
import sys
import tqdm as tqdm

# Define the input and output folders
def get_umap(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load and process each .npy file
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            # Load the data from the .npy file
            file_path = os.path.join(input_folder, file_name)
            data = np.load(file_path, allow_pickle=True)

            print(f'Processing {file_name} with shape {data.shape}')

            # Apply UMAP to scale the features to 2 dimensions
            umap = UMAP(n_components=2)
            vectors = umap.fit_transform(data)

            print(f'Output shape: {vectors.shape}')
            
            # Save the 2-dimensional vectors in the output folder
            output_file_path = os.path.join(output_folder, file_name)
            np.save(output_file_path, vectors)

def get_concate_umap(input_folder1, input_folder2, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load and process each .npy file
    for file_name in os.listdir(input_folder1):
        if file_name.endswith('.npy'):
            # Load the data from the .npy file
            file_path = os.path.join(input_folder1, file_name)
            data1 = np.load(file_path, allow_pickle=True)

            file_path = os.path.join(input_folder2, file_name)
            data2 = np.load(file_path, allow_pickle=True)

            # concatenate the two data
            data = np.concatenate((data1, data2), axis=1)
            # flatten
            data = data.reshape(data.shape[0], -1)

            print(f'Processing {file_name} with shape {data.shape}')
            
            umap = UMAP(n_components=2)
            vectors = umap.fit_transform(data)

            print(f'Output shape: {vectors.shape}')
            
            # Save the 2-dimensional vectors in the output folder
            output_file_path = os.path.join(output_folder, file_name)
            np.save(output_file_path, vectors)

if __name__ == '__main__':
    
    # if there is only one cmd line argument
    if len(sys.argv) == 2:
        parental_folder = sys.argv[1]

        input_folder = os.path.join(parental_folder, 'features')
        output_folder = os.path.join(parental_folder, 'umap')
        get_umap(input_folder, output_folder)

    # if there are two cmd line arguments
    if len(sys.argv) == 4:

        input_folder1 = sys.argv[1]
        input_folder2 = sys.argv[2]
        output_folder = sys.argv[3]

        input_folder1 = os.path.join(input_folder1, 'features')
        input_folder2 = os.path.join(input_folder2, 'features')
        output_folder = os.path.join(output_folder, 'umap')

        get_concate_umap(input_folder1, input_folder2, output_folder)