import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

# Define the path to the folder containing .npy files
#folder_path = '../finetune_model/scratch-sbc5g-pat097-cleanpos/csv'
#save_path =   '../finetune_model/scratch-sbc5g-pat097-cleanpos/count'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]

results = []
result_save_indicator = False

for thresh in tqdm(np.arange(0.01, 1.0, 0.01)):

    # check if this threshold has been calculated
    if os.path.exists(os.path.join(save_path, 'all_ds_prediction_counts_{:.3f}.csv'.format(thresh))):
        continue

    # print("List of datasets to export and infer have been created.")
    with open('list.txt','r') as dataset_file:
        datasets_to_infer = [line.strip().split(".npy")[0] for line in dataset_file.readlines()]

    data = {
        "dataset ID": datasets_to_infer,
        "predicted positive": [-1] * len(datasets_to_infer),
        "predicted negative": [-1] * len(datasets_to_infer),
        "predicted unsure": [-1] * len(datasets_to_infer)
    }
    all_dataset_prediction_counts = pd.DataFrame(data)

    for dataset_id_0 in datasets_to_infer:
        dataset_id = folder_path + '/' + dataset_id_0
        #print(dataset_id_0)

        # USER PARAMETERS (optional)
        unsure_ignored = True 

        # intermediate / output paths
        path_csv_annotations_and_predictions = dataset_id + '.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(path_csv_annotations_and_predictions)

        # Calculate pred_pos, pred_neg, and pred_unsure based on your conditions
        pred_pos = len(df[df['parasite output'] >= thresh])
        pred_neg = len((df['parasite output'] < thresh))
        pred_unsure = len(df) - pred_pos - pred_neg
        all_dataset_prediction_counts.loc[all_dataset_prediction_counts['dataset ID'] == dataset_id_0, ['predicted positive', 'predicted negative', 'predicted unsure']] = [pred_pos, pred_neg, pred_unsure]

    ## add segmentation stats
    df2 = pd.read_csv('utils/cell_count.csv')[['dataset ID', 'Total Count']]
    # Renaming the 'Dataset ID' column in df2 to match the 'dataset ID' column in df1 for consistency
    #df2 = df2.rename(columns={"Dataset ID": "dataset ID"})
    # Merging the datasets on 'dataset ID'
    merged_df = pd.merge(all_dataset_prediction_counts, df2, on="dataset ID")
    # print the columns of df2 and merged_df and all_dataset_prediction_counts
    # Calculating the number of positives per (Total Count / 5e6)
    merged_df['Positives per 5M RBC'] = merged_df['predicted positive'] / (merged_df['Total Count'] / 5e6)

    # save
    merged_df.to_csv(os.path.join(save_path, 'all_ds_prediction_counts_{:.3f}.csv'.format(thresh)))
